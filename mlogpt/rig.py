"""rig.py -- 把一层 transformer 前向自动切分到多台 mlog 处理器，中间值走 memory-cell。

切分规则：语义等于「在一台处理器里逐条算完一层」，超过行数上限的部分自动
分给下一台；跨台的每个中间值 = 1 条 write + 1 条 read。
gen.py 用它生成整张推理网，数值语义与 mlogpt/reference.py 的 numpy 前向一致。
"""
import math

try:                                  # 作为包导入：python mlogpt/gen.py
    from .mschgen import Blueprint, Link
except ImportError:                   # 直接运行本文件时
    from mschgen import Blueprint, Link

import numpy as np


def F(x):
    return "%.17g" % float(x)


def vkey(k):
    return "_".join(str(x) for x in k)


class Unit:
    _uid = [0]

    def __init__(self, name):
        Unit._uid[0] += 1
        self.uid = Unit._uid[0]
        self.name = name
        self.p = "u%d_" % self.uid
        self.stmts = []
        self.uses = set()
        self.usec = {}
        self.produces = []

    def tmp(self, tag):
        return self.p + tag

    def use(self, key):
        self.uses.add(key)
        return "s_" + vkey(key)

    def use_const(self, key, val):
        self.usec[key] = float(val)
        return "s_" + vkey(key)

    def produce(self, key):
        self.produces.append(key)
        return "s_" + vkey(key)

    def add(self, s):
        self.stmts.append(s)


# ---------------------------------------------------------------- 权重
def gen_weights(cfg):
    import numpy as np
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["d"]
    W = {
        "Wq": rng.normal(0, 0.3, (d, d)),
        "Wk": rng.normal(0, 0.3, (d, d)),
        "Wv": rng.normal(0, 0.3, (d, d)),
        "Wo": rng.normal(0, 0.3, (d, d)),
        "W1": rng.normal(0, 0.3, (4 * d, d)),
        "W2": rng.normal(0, 0.3, (d, 4 * d)),
    }
    g = rng.normal(1, 0.05, d)
    b = rng.normal(0, 0.05, d)
    X = rng.normal(0, 1, (cfg["S"], d))
    return W, g, b, X


# ---------------------------------------------------------------- matvec 行
def mv_units(li, name, M, in_src, max_seg=256):
    rows, cols = len(M), len(M[0])
    units = []
    for r in range(rows):
        nseg = (cols + max_seg - 1) // max_seg
        for si in range(nseg):
            seg = list(range(si * max_seg, min((si + 1) * max_seg, cols)))
            u = Unit("%s_%d_s%d" % (name, r, si))
            iv = {c: in_src(li, c, u) for c in seg}
            last = si == nseg - 1
            outkey = (li, name, r)
            partkey = (li, name, r, "part", si)
            if si == 0:
                acc = u.produce(outkey) if last else u.tmp("acc")
                u.add("op mul %s %s %s" % (acc, F(M[r][seg[0]]), iv[seg[0]]))
            else:
                acc = u.tmp("acc")
                u.add("op mul %s %s %s" % (acc, F(M[r][seg[0]]), iv[seg[0]]))
                pv = u.use((li, name, r, "part", si - 1))
                u.add("op add %s %s %s" % (acc, acc, pv))
            for c in seg[1:]:
                t = u.tmp("t%d" % c)
                u.add("op mul %s %s %s" % (t, F(M[r][c]), iv[c]))
                u.add("op add %s %s %s" % (acc, acc, t))
            if not last:
                u.add("set %s %s" % (u.produce(partkey), acc))
            elif si > 0:
                u.add("set %s %s" % (u.produce(outkey), acc))
            units.append(u)
    return units


# ---------------------------------------------------------------- 一层
def layer_units(cfg, li, W, g, b, X):
    d, S = cfg["d"], cfg["S"]
    units = []
    sqrt_d = math.sqrt(d)

    def last_ctx(li_, c, u):
        if li_ == 0:
            return u.use_const((li_, "xlast", c), X[S - 1][c])
        return u.use((li_ - 1, "out", c))

    def ctx(j):
        def src(li_, c, u):
            if li_ == 0 or j < S - 1:
                return u.use_const((li_, "x", j, c), X[j][c])
            return u.use((li_ - 1, "out", c))
        return src

    # q = Wq * h_last
    units += mv_units(li, "q", W["Wq"], last_ctx)
    # k_j, v_j, s_j
    for j in range(S):
        units += mv_units(li, "k%d" % j, W["Wk"], ctx(j))
        units += mv_units(li, "v%d" % j, W["Wv"], ctx(j))
        u = Unit("score_%d" % j)
        acc = u.produce((li, "s", j))
        u.add("op mul %s %s %s" % (acc, u.use((li, "q", 0)), u.use((li, "k%d" % j, 0))))
        for i in range(1, d):
            t = u.tmp("t%d" % i)
            u.add("op mul %s %s %s" % (t, u.use((li, "q", i)), u.use((li, "k%d" % j, i))))
            u.add("op add %s %s %s" % (acc, acc, t))
        u.add("op div %s %s %s" % (acc, acc, F(sqrt_d)))
        units.append(u)
    # softmax
    u = Unit("softmax")
    mx = u.tmp("mx")
    u.add("set %s %s" % (mx, u.use((li, "s", 0))))
    for j in range(1, S):
        u.add("op max %s %s %s" % (mx, mx, u.use((li, "s", j))))
    ssum = u.tmp("sum")
    e = {}
    for j in range(S):
        dj = u.tmp("d%d" % j)
        u.add("op sub %s %s %s" % (dj, u.use((li, "s", j)), mx))
        e[j] = u.tmp("e%d" % j)
        u.add("op pow %s 2.718281828459045 %s" % (e[j], dj))
    u.add("set %s %s" % (ssum, e[0]))
    for j in range(1, S):
        u.add("op add %s %s %s" % (ssum, ssum, e[j]))
    for j in range(S):
        u.add("op div %s %s %s" % (u.produce((li, "p", j)), e[j], ssum))
    units.append(u)
    # a_i = sum_j p_j v_j_i
    for i in range(d):
        u = Unit("attn_%d" % i)
        acc = u.produce((li, "a", i))
        u.add("op mul %s %s %s" % (acc, u.use((li, "p", 0)), u.use((li, "v0", i))))
        for j in range(1, S):
            t = u.tmp("t%d" % j)
            u.add("op mul %s %s %s" % (t, u.use((li, "p", j)), u.use((li, "v%d" % j, i))))
            u.add("op add %s %s %s" % (acc, acc, t))
        units.append(u)
    # o = Wo * a
    units += mv_units(li, "o", W["Wo"], lambda li_, c, u: u.use((li_, "a", c)))
    # h = x_last + o ; LN
    u = Unit("ln_h_mu")
    h = {}
    for i in range(d):
        oi = u.use((li, "o", i))
        xi = last_ctx(li, i, u)
        h[i] = u.tmp("h%d" % i)
        u.add("op add %s %s %s" % (h[i], xi, oi))
    mu = u.tmp("mu")
    u.add("set %s %s" % (mu, h[0]))
    for i in range(1, d):
        u.add("op add %s %s %s" % (mu, mu, h[i]))
    u.add("op div %s %s %s" % (mu, mu, F(d)))
    for i in range(d):
        u.add("set %s %s" % (u.produce((li, "h", i)), h[i]))
    u.add("set %s %s" % (u.produce((li, "mu",)), mu))
    units.append(u)

    u = Unit("ln_var")
    mu2 = u.use((li, "mu",))
    var = u.tmp("var")
    for i in range(d):
        ci = u.tmp("c%d" % i)
        u.add("op sub %s %s %s" % (ci, u.use((li, "h", i)), mu2))
        c2 = u.tmp("c2_%d" % i)
        u.add("op mul %s %s %s" % (c2, ci, ci))
        if i == 0:
            u.add("set %s %s" % (var, c2))
        else:
            u.add("op add %s %s %s" % (var, var, c2))
    u.add("op div %s %s %s" % (var, var, F(d)))
    u.add("op add %s %s 0.00001" % (var, var))
    rs = u.tmp("rs")
    u.add("op pow %s %s -0.5" % (rs, var))
    u.add("set %s %s" % (u.produce((li, "rs",)), rs))
    units.append(u)

    # n_i = (h_i - mu) * rs * g_i + b_i，按 64 行一组切（每行 4 条 + 2 个常量）
    for g0 in range(0, d, 64):
        u = Unit("ln_n_%d" % g0)
        mu3 = u.use((li, "mu",))
        rs2 = u.use((li, "rs",))
        for i in range(g0, min(g0 + 64, d)):
            ci = u.tmp("c%d" % i)
            u.add("op sub %s %s %s" % (ci, u.use((li, "h", i)), mu3))
            mi = u.tmp("m%d" % i)
            u.add("op mul %s %s %s" % (mi, ci, rs2))
            gi = u.use_const((li, "g", i), g[i])
            ni = u.produce((li, "n", i))
            u.add("op mul %s %s %s" % (ni, mi, gi))
            bi = u.use_const((li, "b", i), b[i])
            u.add("op add %s %s %s" % (ni, ni, bi))
        units.append(u)

    # MLP1: u = relu(W1 n)
    mlp1 = mv_units(li, "u", W["W1"], lambda li_, c, u: u.use((li_, "n", c)))
    for u in mlp1:
        for k in list(u.produces):
            u.stmts.append("op max %s %s 0" % ("s_" + vkey(k), "s_" + vkey(k)))
    units += mlp1
    # MLP2: y = W2 u
    units += mv_units(li, "y", W["W2"], lambda li_, c, u: u.use((li_, "u", c)))
    # out = n + y
    u = Unit("out")
    for i in range(d):
        if li == cfg["L"] - 1:
            u.add("op add out%d %s %s" % (i, u.use((li, "n", i)), u.use((li, "y", i))))
            u.final = i
        else:
            u.add("op add %s %s %s" % (u.produce((li, "out", i)),
                                       u.use((li, "n", i)), u.use((li, "y", i))))
    units.append(u)
    return units


# ---------------------------------------------------------------- 打包
def emit_chunk(chunk, slotmap, cellname, needed_outside):
    consts = {}
    reads = []
    produced = set()
    for u in chunk:
        consts.update(u.usec)
        produced.update(u.produces)
    seen = set()
    for u in chunk:
        for k in sorted(u.uses):
            if k in produced or k in seen:
                continue
            seen.add(k)
            reads.append(k)
    lines = []
    for k in sorted(consts, key=vkey):
        lines.append("set %s %s" % ("s_" + vkey(k), F(consts[k])))
    for k in reads:
        c, s = slotmap[k]
        lines.append("read %s %s %d" % ("s_" + vkey(k), cellname(c), s))
    for u in chunk:
        lines.extend(u.stmts)
    for u in chunk:
        for k in u.produces:
            if k in needed_outside:
                c, s = slotmap[k]
                lines.append("write %s %s %d" % ("s_" + vkey(k), cellname(c), s))
    return lines


def pack(units, cap):
    def est(chunk):
        return len(emit_chunk(chunk, _dummy_slots(chunk), lambda c: "c",
                              set(k for u in chunk for k in u.produces)))

    chunks = []
    cur = []
    for u in units:
        cur.append(u)
        if est(cur) > cap:
            cur.pop()
            if not cur:
                raise SystemExit("unit %s alone exceeds cap %d (%d stmts)"
                                 % (u.name, cap, len(u.stmts)))
            chunks.append(cur)
            cur = [u]
            if est(cur) > cap:
                raise SystemExit("unit %s alone exceeds cap %d" % (u.name, cap))
    if cur:
        chunks.append(cur)
    return chunks


def _dummy_slots(chunk):
    m = {}
    n = 0
    for u in chunk:
        for k in u.produces:
            if k not in m:
                m[k] = (0, n)
                n += 1
    for u in chunk:
        for k in u.uses:
            if k not in m:
                m[k] = (0, n)
                n += 1
    return m


class Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.g, self.b, self.X = gen_weights(cfg)
        self.units = []
        for li in range(cfg["L"]):
            self.units += layer_units(cfg, li, self.W, self.g, self.b, self.X)

    def ref(self):
        cfg = self.cfg
        d, S, L = cfg["d"], cfg["S"], cfg["L"]
        W, g, b, X = self.W, self.g, self.b, self.X
        h = X[S - 1].copy()
        for li in range(L):
            ctx = np.stack([X[j] for j in range(S - 1)] + [h])
            K = W["Wk"] @ ctx.T
            V = W["Wv"] @ ctx.T
            q = W["Wq"] @ h
            s = (K.T @ q) / math.sqrt(d)
            p = np.exp(s - s.max())
            p = p / p.sum()
            a = p @ V.T
            o = W["Wo"] @ a
            hh = h + o
            mu = hh.mean()
            var = ((hh - mu) ** 2).mean()
            n = (hh - mu) / math.sqrt(var + 1e-5) * g + b
            u = np.maximum(W["W1"] @ n, 0.0)
            y = W["W2"] @ u
            h = n + y
        return h

    def build(self, cap):
        chunks = pack(self.units, cap)
        # 哪些 key 需要跨块写 cell
        chunk_of = {}
        for ci, ch in enumerate(chunks):
            for u in ch:
                for k in u.produces:
                    chunk_of[k] = ci
        needed = set()
        for ci, ch in enumerate(chunks):
            for u in ch:
                for k in u.uses:
                    if chunk_of.get(k) != ci:
                        needed.add(k)
        # slot 分配（按产出顺序）
        slotmap = {}
        order = []
        for ch in chunks:
            for u in ch:
                for k in u.produces:
                    if k not in slotmap and k in needed:
                        slotmap[k] = None
                        order.append(k)
        slots_per_cell = self.cfg.get("slots_per_cell", 64)
        for i, k in enumerate(order):
            slotmap[k] = (i // slots_per_cell, i % slots_per_cell)
        self.slotmap = slotmap
        self.chunks = chunks
        self.codes = []
        for ch in chunks:
            self.codes.append(emit_chunk(ch, slotmap, lambda c: "cell%d" % (c + 1), needed))
        return self.codes

    def stats(self):
        ncell = 0
        if self.slotmap:
            ncell = max(c for c, s in self.slotmap.values()) + 1
        total = sum(len(c) for c in self.codes)
        body = sum(len(u.stmts) for u in self.units)
        io = total - body
        return dict(processors=len(self.codes), cells=ncell, slots=len(self.slotmap),
                    body_stmts=body, total_stmts=total, io_stmts=io,
                    io_ratio=io / total if total else 0.0,
                    max_stmts=max(len(c) for c in self.codes),
                    per_proc=[len(c) for c in self.codes])


BLOCK_RANGE = {"micro-processor": 10.0, "logic-processor": 22.0, "hyper-processor": 42.0}


def layout(nproc, ncell):
    """紧凑布局：cell 网格在左上，处理器网格紧贴其下；返回 (cellpos, procpos, W)。"""
    if ncell == 0:
        cw = 0
        cell_rows = 0
    else:
        cw = max(1, int(math.ceil(math.sqrt(ncell))))
        cell_rows = (ncell + cw - 1) // cw
    cellpos = {}
    for ci in range(ncell):
        cellpos[ci] = (ci % cw, ci // cw)
    W = max(1, int(math.ceil(math.sqrt(max(1, nproc)))))
    y0 = cell_rows + 1 if ncell else 0
    procpos = [(i % W, y0 + i // W) for i in range(nproc)]
    return cellpos, procpos, W


def build_msch(rig, path, block=None, name="rig"):
    codes = rig.codes
    ncell = 0
    if rig.slotmap:
        ncell = max(c for c, s in rig.slotmap.values()) + 1
    cellpos, procpos, W = layout(len(codes), ncell)
    linkspec = []
    maxd = 0.0
    for i, code in enumerate(codes):
        px, py = procpos[i]
        used_cells = set(int(line.split()[2][4:]) - 1 for line in code
                         if line.split() and line.split()[0] in ("read", "write"))
        ls = []
        for c in sorted(used_cells):
            cx, cy = cellpos[c]
            dx, dy = cx - px, cy - py
            maxd = max(maxd, math.hypot(dx, dy))
            ls.append((c, dx, dy))
        linkspec.append(ls)
    if block is None:
        block = "hyper-processor"
        for b in ("micro-processor", "logic-processor", "hyper-processor"):
            if maxd <= BLOCK_RANGE[b]:
                block = b
                break
    feasible = maxd <= BLOCK_RANGE[block]
    bp = Blueprint(name=name, description="generated by mlogpt/rig.py", labels=[])
    for ci in range(ncell):
        bp.add_block("memory-cell", cellpos[ci][0], cellpos[ci][1])
    for i, code in enumerate(codes):
        px, py = procpos[i]
        links = [Link("cell%d" % (c + 1), dx, dy) for c, dx, dy in linkspec[i]]
        bp.add_logic("\n".join(code) + "\n", px, py, block=block, links=links)
    data = bp.build()
    open(path, "wb").write(data)
    return dict(path=path, bytes=len(data), bounds=bp.bounds(), ncell=ncell,
                max_link_dist=maxd, nproc=len(codes), block=block,
                range=BLOCK_RANGE[block], feasible=feasible, grid_w=W)
