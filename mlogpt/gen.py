# -*- coding: utf-8 -*-
"""gen.py -- 把一个小 transformer 展开成 Mindustry mlog 蓝图（.msch）。

结构（全部 world-processor + world-cell + message）：
  1  init        ：等 loader 屏障 -> 写种子 token / r 表 / INIT=1
  nloader        ：把嵌入表 E (V x d) 以字面量写进 world-cell（每台 1 个 cell = 512 值）
  nlvl1          ：loader 屏障的两级求和
  S  preload     ：每轮按 GO_A 相位，用 getlink 动态选嵌入 cell，把 E[tok_j] 搬进 arena（S*d 槽）
  nfwd  forward  ：前向块（层 0 的 X 全部改成读 arena 的静态槽），依赖 flag 轮询 + GO_B 相位
  nhead head     ：V 行 logits = Wout*out，每台 3 行；块内先算局部 exp 和，等 THR_PH 后
                   只有 TARGET 块做块内累积搜索
  1  prefix      ：累加 tot -> thr = r[last]*tot -> 找 TARGET/OFFSET -> 写 THR_PH
  1  driver      ：相位 GO_A/GO_B、窗口左移、C2_OUT、CNT
  1  printer     ：把新 token 打到 message1（每 4 个换行）

数值语义 = mlogpt/reference.py 的 numpy 实现（同权重、同序运算）；stats.json
里的 ref_tokens 就是它的输出，可直接和游戏内跑出来的序列对拍。

用法：
  构建：python mlogpt/gen.py --build --d 128 --S 16 --L 4 --V 4096 --N 8 \\
            --cap 900 --seed-tok 82 --table-free --input --input-text 量持法 \\
            --weights weights/mlogpt-zh-0.72m-fp32.npz \\
            --vocab weights/vocab-zh-4096.json --out mlogpt-zh-0.72m.msch
  运行：python mlogpt/gen.py --run mlogpt-zh-0.72m.msch --ticks 120 \\
            --input-text 量持法        # 需 Mindustry v160.7 jar + 姊妹仓的模拟器
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))          # 仓库内的 mlogpt/
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))      # 仓库根
for _p in (HERE, ROOT):                                    # 只加仓库自身
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:                                                       # python -m mlogpt.gen
    from .mschgen import Blueprint, Link
    from . import rig as base
    from .rig_chain import mv_rows, dot_rows, reduce_keys, _final, Unit, F, vkey
    from .reference import (LN_EPS, emb_params, emb_row, load_weights, peek_shapes,
                            ref_forward, sample_ref, text_to_ids)
except ImportError:                                        # python mlogpt/gen.py
    from mlogpt.mschgen import Blueprint, Link
    from mlogpt import rig as base
    from mlogpt.rig_chain import mv_rows, dot_rows, reduce_keys, _final, Unit, F, vkey
    from mlogpt.reference import (LN_EPS, emb_params, emb_row, load_weights, peek_shapes,
                                  ref_forward, sample_ref, text_to_ids)

DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "mlogpt-zh-0.72m-fp32.npz")
DEFAULT_VOCAB = os.path.join(ROOT, "weights", "vocab-zh-4096.json")

LIMIT = 1000        # 单台处理器最多 1000 条；1001 条起静默丢弃
SLOTS = 512         # world-cell 槽数
E7 = 2.718281828459045
LN_EPS = 0.00001
RAD2DEG = 180.0 / math.pi   # mlog 的 op sin 吃角度（度）

# ---- 控制格 sp0 的槽位
SL_TOK = 0          # 0..S-1   窗口 token
SL_R = 32           # 32..     r 表（N 个）
SL_OUT = 64         # 64..     已生成 token
SL_CNT = 128
SL_INIT = 129
SL_GO_A = 130
SL_GO_B = 131
SL_DONE_C = 132
SL_TOKOUT = 133
SL_THR = 134
SL_TARGET = 135
SL_OFFSET = 136
SL_THR_PH = 137
SL_SCAN_PH = 138   # 组内扫描已定出目标 head 的相位
SL_HEAD = 139      # 目标 head 下标（与 SL_TARGET 的组号分开，否则组号与 head 号会撞）
SL_IN = 140        # 140..    玩家输入的 S 个 token（分词器写 / driver 读）
# ---- 游戏内状态面板（reporter 读、init/driver 写）
SL_PROG = 200      # 数字嵌入载入进度（已载入行数）
SL_PROGN = 201     # 需要载入的总行数
SL_PHASE = 202     # 0 载入中 / 1 就绪等输入 / 2 生成中 / 3 完成
SL_GEN = 203       # 本轮已生成 token 数
SL_HB = 204        # driver 心跳（等输入/生成时自增，用来区分「等输入」与「driver 死了」）


# ---------------------------------------------------------------- 小工具
class Asm:
    """按行号解析 jump 目标的汇编器（mlog 的 jump 目标是 0 基行号）。"""

    def __init__(self):
        self.lines = []
        self.lab = {}

    def add(self, s):
        self.lines.append(s)
        return len(self.lines) - 1

    def label(self, n):
        self.lab[n] = len(self.lines)

    def jmp(self, lab, cond="always", a="0", b="0"):
        self.lines.append(("J", lab, cond, a, b))

    def render(self):
        out = []
        for it in self.lines:
            if isinstance(it, tuple):
                _, lab, cond, a, b = it
                out.append("jump %d %s %s %s" % (self.lab[lab], cond, a, b))
            else:
                out.append(it)
        for ln in out:
            assert ln.strip(), "empty mlog line"
        return out


def emit_chunk(chunk, slotmap, cellname, needed_outside):
    """与 rig.py 的 emit_chunk 同语义，只是排序用 vkey（避免异构元组比较）。"""
    consts, produced, seen, reads = {}, set(), set(), []
    for u in chunk:
        consts.update(u.usec)
        produced.update(u.produces)
    for u in chunk:
        for k in sorted(u.uses, key=vkey):
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


def chunk_len(ch, needed, ndeps, nxdeps, nwrite_extra=0):
    """分析式行数（= emit_chunk 的长度 + 包装语句），避免反复 emit。"""
    consts, produced, uses = {}, set(), set()
    nstmts = 0
    for u in ch:
        consts.update(u.usec)
        produced.update(u.produces)
        uses.update(u.uses)
        nstmts += len(u.stmts)
    reads = len(uses - produced)
    writes = sum(1 for u in ch for k in u.produces if k in needed)
    # setrate + init0(2) + set last + L(2) + deps*2 + xdeps*2 + write flag + set last + jump
    wrap = 1 + 2 + 1 + 2 + 2 * ndeps + 2 * nxdeps + 2 + 1
    return wrap + len(consts) + reads + nstmts + writes + nwrite_extra


def pack_units(units, cap_pack):
    """贪心打包；写入数按「全部 produced」保守估计。"""
    chunks, cur = [], []
    consts, produced, ext = set(), set(), set()
    nstmts = 0
    for u in units:
        uprod = set(u.produces)
        nconst = len(consts | set(u.usec))
        tmp = set()
        for k in u.uses:
            if k not in produced and k not in ext:
                tmp.add(k)
        n_ext = len(ext) + len(tmp) - len((ext | tmp) & uprod)
        size = nconst + n_ext + nstmts + len(u.stmts) + len(produced | uprod)
        if cur and size > cap_pack:
            chunks.append(cur)
            cur, consts, produced, ext, nstmts = [], set(), set(), set(), 0
            nconst = len(set(u.usec))
            tmp = set(k for k in u.uses)
            n_ext = len(tmp)
            size = nconst + n_ext + len(u.stmts) + len(uprod)
        cur.append(u)
        consts |= set(u.usec)
        ext |= tmp
        produced |= uprod
        ext -= uprod
        nstmts += len(u.stmts)
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- 权重
def _saychar(ch):
    """print 用的字符安全化：'+unk 占位/多字符/引号/反斜杠/空白控制符 一律降级成 ?。"""
    if not ch or len(ch) != 1:
        return "?"
    if ch in ('"', "\\", "\n", "\r", "\t"):
        return "?"
    return ch


# ---------------------------------------------------------------- 前向 units（动态 X）
def layer_units_dyn(cfg, li, W, g, b):
    """= rig_chain.layer_units_chain，但 layer-0 / 非最后位置的 X 源改成外部键
    (0,"X",j,c)（由 preload 每轮搬进 arena），并且最后一层也产出 (li,"out",i) 键。"""
    d, S = cfg["d"], cfg["S"]
    units = []
    sqrt_d = math.sqrt(d)

    def last_src(c):
        if li == 0:
            return ("key", (0, "X", S - 1, c))
        return ("key", (li - 1, "out", c))

    def mk_src(j):
        def src(c):
            if li == 0 or j < S - 1:
                return ("key", (0, "X", j, c))
            return ("key", (li - 1, "out", c))
        return src

    mv_rows(units, li, "q", W["Wq"], last_src)
    for j in range(S):
        mv_rows(units, li, "k%d" % j, W["Wk"], mk_src(j))
        mv_rows(units, li, "v%d" % j, W["Wv"], mk_src(j))

    def score_terms(j):
        return [("key", (li, "q", i), "key", (li, "k%d" % j, i)) for i in range(d)]
    dot_rows(units, li, "s", S, score_terms, post=["op div {key} {key} " + F(sqrt_d)])

    def _mx_in(u):
        cur = [u.use((li, "s", j)) for j in range(S)]
        lvl = 0
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur) - 1, 2):
                t = u.tmp("mx%d_%d" % (lvl, i))
                u.add("op max %s %s %s" % (t, cur[i], cur[i + 1]))
                nxt.append(t)
            if len(cur) % 2:
                nxt.append(cur[-1])
            cur = nxt
            lvl += 1
        return cur[0]

    GROUP = 8
    for g0 in range(0, S, GROUP):
        v = Unit("softmax_exp_%d" % g0)
        m = _mx_in(v)
        for j in range(g0, min(g0 + GROUP, S)):
            dj = v.tmp("d%d" % j)
            v.add("op sub %s %s %s" % (dj, v.use((li, "s", j)), m))
            v.add("op pow %s 2.718281828459045 %s" % (v.produce((li, "e", j)), dj))
        units.append(v)
    esum = reduce_keys(units, li, [(li, "e", j) for j in range(S)], "softmax_sum")
    for g0 in range(0, S, GROUP):
        v = Unit("softmax_p_%d" % g0)
        ss = v.use(esum)
        for j in range(g0, min(g0 + GROUP, S)):
            v.add("op div %s %s %s" % (v.produce((li, "p", j)), v.use((li, "e", j)), ss))
        units.append(v)

    def attn_terms(i):
        return [("key", (li, "p", j), "key", (li, "v%d" % j, i)) for j in range(S)]
    dot_rows(units, li, "a", d, attn_terms)

    mv_rows(units, li, "o", W["Wo"], lambda c: ("key", (li, "a", c)))

    for g0 in range(0, d, GROUP):
        u = Unit("ln_h_%d" % g0)
        for i in range(g0, min(g0 + GROUP, d)):
            if li == 0:
                xi = u.use((0, "X", S - 1, i))
            else:
                xi = u.use((li - 1, "out", i))
            u.add("op add %s %s %s" % (u.produce((li, "h", i)), xi, u.use((li, "o", i))))
        units.append(u)
    hsum = reduce_keys(units, li, [(li, "h", i) for i in range(d)], "ln_mu")
    _final(units, "ln_mu_fin", (li, "mu"), hsum, ["op div {acc} {src} " + F(d)])
    for g0 in range(0, d, GROUP):
        u = Unit("ln_c2_%d" % g0)
        mu2 = u.use((li, "mu"))
        for i in range(g0, min(g0 + GROUP, d)):
            ci = u.tmp("c%d" % i)
            u.add("op sub %s %s %s" % (ci, u.use((li, "h", i)), mu2))
            u.add("op mul %s %s %s" % (u.produce((li, "c2", i)), ci, ci))
        units.append(u)
    c2sum = reduce_keys(units, li, [(li, "c2", i) for i in range(d)], "ln_var")
    _final(units, "ln_rs_fin", (li, "rs"), c2sum,
           ["op div {acc} {src} " + F(d),
            "op add {acc} {acc} " + F(LN_EPS),
            "op pow {key} {acc} -0.5"])
    for g0 in range(0, d, GROUP):
        u = Unit("ln_n_%d" % g0)
        mu3, rs2 = u.use((li, "mu")), u.use((li, "rs"))
        for i in range(g0, min(g0 + GROUP, d)):
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

    mv_rows(units, li, "u", W["W1"], lambda c: ("key", (li, "n", c)),
            post=["op max {key} {key} 0"])
    mv_rows(units, li, "y", W["W2"], lambda c: ("key", (li, "u", c)))
    for g0 in range(0, d, GROUP):
        u = Unit("out_%d" % g0)
        for i in range(g0, min(g0 + GROUP, d)):
            u.add("op add %s %s %s" % (u.produce((li, "out", i)),
                                       u.use((li, "n", i)), u.use((li, "y", i))))
        units.append(u)
    return units


# ---------------------------------------------------------------- numpy 参考
# ---------------------------------------------------------------- 构建
def build(cfg, W, path, stats_path=None, verbose=True):
    t0 = time.time()
    d, S, L, V, N = cfg["d"], cfg["S"], cfg["L"], cfg["V"], cfg["N"]
    cap, TEMP = cfg["cap"], cfg["TEMP"]
    assert SLOTS % d == 0, "d must divide %d" % SLOTS
    rows_per_cell = SLOTS // d
    nacell = (S * d + SLOTS - 1) // SLOTS
    assert N <= 32 and S <= 32, "S,N <= 32"

    # ---- message 方块下标的唯一口径（0 基）：输出 N 个 -> 说明 -> 状态 ->（输入模式）输入
    voc = cfg.get("vocab") or []
    nout = N if voc else 0
    MSG_INSTR = nout
    MSG_STAT = nout + 1
    MSG_IN = (nout + 2) if (cfg.get("input") and voc) else None
    # 说明/状态面板：只对 --vocab（say 模式，即游戏内可用版）加
    LOG_PANEL = bool(voc)

    # ---- 前向 units -> 块
    units = []
    for li in range(L):
        units += layer_units_dyn(cfg, li, W, W["g"], W["b"])
    if verbose:
        print("W\tunits %d\t%.1fs" % (len(units), time.time() - t0), flush=True)
    chunks = pack_units(units, cap)
    if verbose:
        print("W\tpacked %d chunks\t%.1fs" % (len(chunks), time.time() - t0), flush=True)

    def analyze(chs):
        chunk_of = {}
        for ci, ch in enumerate(chs):
            for u in ch:
                for k in u.produces:
                    chunk_of[k] = ci
        needed, deps, xdeps = set(), [], []
        for ci, ch in enumerate(chs):
            dp, xd = set(), set()
            for u in ch:
                for k in u.uses:
                    if k[0] == 0 and k[1] == "X":
                        xd.add(k[2])
                        continue
                    cj = chunk_of.get(k)
                    if cj is None:
                        raise SystemExit("use without producer: %s" % (k,))
                    if cj != ci:
                        needed.add(k)
                        dp.add(cj)
            deps.append(dp)
            xdeps.append(xd)
        return chunk_of, needed, deps, xdeps

    nsplit = 0
    for rnd in range(8):
        chunk_of, needed, deps, xdeps = analyze(chunks)
        over = [ci for ci, ch in enumerate(chunks)
                if chunk_len(ch, needed, len(deps[ci]), len(xdeps[ci])) > LIMIT]
        if not over:
            break
        for ci in sorted(over, reverse=True):
            ch = chunks[ci]
            if len(ch) < 2:
                raise SystemExit("single-unit chunk over limit (%d), lower --cap"
                                 % len(ch))
            k = len(ch) // 2
            chunks[ci:ci + 1] = [ch[:k], ch[k:]]
            nsplit += 1
        if verbose:
            print("W\tsplit %d chunks (round %d) -> %d\t%.1fs"
                  % (len(over), rnd, len(chunks), time.time() - t0), flush=True)
    chunk_of, needed, deps, xdeps = analyze(chunks)
    # 最后一层的 out 一定要落盘（head 读它，但没有别的块消费它 -> 不在 needed 里）
    needed |= set((L - 1, "out", i) for i in range(d))
    nfwd = len(chunks)
    lens = [chunk_len(ch, needed, len(deps[ci]), len(xdeps[ci])) for ci, ch in enumerate(chunks)]
    if max(lens) > LIMIT:
        raise SystemExit("chunk still over %d after split" % LIMIT)

    # ---- slotmap：arena 占 0..nacell-1，其余跨块键依次排
    slotmap = {}
    for j in range(S):
        for c in range(d):
            i = j * d + c
            slotmap[(0, "X", j, c)] = (i // SLOTS, i % SLOTS)
    order = sorted(needed, key=vkey)
    for i, k in enumerate(order):
        slotmap[k] = (nacell + i // SLOTS, i % SLOTS)
    nvalcell = (len(order) + SLOTS - 1) // SLOTS
    ne = nacell + nvalcell

    # ---- 单元名
    cellnames = ["cell%d" % (i + 1) for i in range(ne)]
    tf = bool(cfg.get("table_free"))
    nemb = 0 if tf else (V + rows_per_cell - 1) // rows_per_cell
    embnames = ["emb%d" % (i + 1) for i in range(nemb)]
    hr = cfg["head_rows"]
    nhead = (V + hr - 1) // hr
    nffc = (nfwd + SLOTS - 1) // SLOTS
    nhfc = (nhead + SLOTS - 1) // SLOTS
    nloader = nemb
    nlv1 = (nloader + 127) // 128
    nlfc = (nloader + SLOTS - 1) // SLOTS
    specials = (["sp0", "pf0", "ls0", "gs0", "gf0"] + ["lf%d" % i for i in range(nlfc)]
                + ["ff%d" % i for i in range(nffc)] + ["hf%d" % i for i in range(nhfc)]
                + ["pt%d" % i for i in range(nhfc)]
                + ["pw%d" % i for i in range((V + SLOTS - 1) // SLOTS)]
                + ["bs0", "bf0"])
    nptc = nhfc
    allcells = cellnames + embnames + specials

    procs = []          # (name, lines, first_links)

    # ---- init
    a = Asm()
    a.add("setrate 1000")
    a.label("i0")
    a.add("read f sp0 %d" % SL_INIT)
    a.jmp("halt", "equal", "f", "1")
    a.add("set s 0")
    a.add("write %d sp0 %d" % (nlv1, SL_PROGN))
    for i in range(nlv1):
        a.add("read lv ls0 %d" % i)
        a.add("op add s s lv")
        a.add("write s sp0 %d" % SL_PROG)
    a.jmp("i0", "lessThan", "s", str(nlv1))
    if not cfg.get("input"):
        for j in range(S):
            a.add("write %d sp0 %d" % (cfg["seed_tok"], SL_TOK + j))
    for t in range(N):
        a.add("write %s sp0 %d" % (F(W["r"][t]), SL_R + t))
    a.add("write 1 sp0 %d" % SL_INIT)
    a.add("write 1 sp0 %d" % SL_PHASE)
    a.label("halt")
    a.add("wait 0")
    a.jmp("halt")
    procs.append(("init", a.render(), []))

    # ---- loader（每台 1 个 cell）与两级屏障
    for ci in range(nloader):
        a = Asm()
        a.add("setrate 1000")
        for r in range(rows_per_cell):
            tok = ci * rows_per_cell + r
            if tok >= V:
                continue
            for c in range(d):
                a.add("write %s emb%d %d" % (F(W["E"][tok][c]), ci + 1, r * d + c))
        a.add("write 1 lf%d %d" % (ci // SLOTS, ci % SLOTS))
        a.label("halt")
        a.add("wait 0")
        a.jmp("halt")
        procs.append(("loader%d" % ci, a.render(), []))
    for li_ in range(nlv1):
        a = Asm()
        a.add("setrate 1000")
        lo, hi = li_ * 128, min(nloader, (li_ + 1) * 128)
        a.label("L")
        a.add("set s 0")
        for i in range(lo, hi):
            a.add("read lf lf%d %d" % (i // SLOTS, i % SLOTS))
            a.add("op add s s lf")
        a.jmp("L", "lessThan", "s", str(hi - lo))
        a.add("write 1 ls0 %d" % li_)
        a.label("halt")
        a.add("wait 0")
        a.jmp("halt")
        procs.append(("lv1_%d" % li_, a.render(), []))

    # ---- preload（每位置一台）。有表：getlink 动态选嵌入 cell；无表：游戏内算 sin 特征
    ea, eb = emb_params(d)
    for j in range(S):
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read g sp0 %d" % SL_GO_A)
        a.jmp("L", "lessThanEq", "g", "last")
        a.add("read tk sp0 %d" % (SL_TOK + j))
        if tf:
            # X[t][c] = sin(a_c*t + b_c)（mlog 的 sin 吃角度，故先乘 180/pi）
            for c in range(d):
                i = j * d + c
                a.add("op mul e %s tk" % F(ea[c] * RAD2DEG))
                a.add("op add e e %s" % F(eb[c] * RAD2DEG))
                a.add("op sin e e")
                a.add("write e cell%d %d" % (i // SLOTS + 1, i % SLOTS))
        else:
            a.add("op div ci tk %d" % rows_per_cell)
            a.add("op mod m tk %d" % rows_per_cell)
            a.add("op mul si m %d" % d)
            a.add("getlink lnk ci")
            for c in range(d):
                i = j * d + c
                a.add("op add ad si %d" % c)
                a.add("read xv lnk ad")
                a.add("write xv cell%d %d" % (i // SLOTS + 1, i % SLOTS))
        a.add("write g pf0 %d" % j)
        a.add("set last g")
        a.jmp("L")
        first_links = [] if tf else list(embnames)
        procs.append(("preload%d" % j, a.render(), first_links))

    # ---- forward
    for ci, ch in enumerate(chunks):
        body = emit_chunk(ch, slotmap, lambda c: "cell%d" % (c + 1), needed)
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read ph sp0 %d" % SL_GO_B)
        a.jmp("L", "lessThanEq", "ph", "last")
        for p in sorted(deps[ci]):
            a.add("read wf ff%d %d" % (p // SLOTS, p % SLOTS))
            a.jmp("L", "lessThan", "wf", "ph")
        for j in sorted(xdeps[ci]):
            a.add("read wf pf0 %d" % j)
            a.jmp("L", "lessThan", "wf", "ph")
        for st in body:
            a.add(st)
        a.add("write ph ff%d %d" % (ci // SLOTS, ci % SLOTS))
        a.add("set last ph")
        a.jmp("L")
        procs.append(("fwd%d" % ci, a.render(), []))

    # ---- head
    outkeys = [(L - 1, "out", i) for i in range(d)]
    outchunks = sorted(set(chunk_of[k] for k in outkeys))
    for hi in range(nhead):
        rows = [(r, W["Wout"][r]) for r in range(hi * hr, min(V, (hi + 1) * hr))]
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read ph sp0 %d" % SL_GO_B)
        a.jmp("L", "lessThanEq", "ph", "last")
        for p in outchunks:
            a.add("read wf ff%d %d" % (p // SLOTS, p % SLOTS))
            a.jmp("L", "lessThan", "wf", "ph")
        for i in range(d):
            c, s = slotmap[outkeys[i]]
            a.add("read h%d cell%d %d" % (i, c + 1, s))
        for (r, Wr) in rows:
            a.add("op mul lt%d %s h0" % (r, F(Wr[0])))
            for c in range(1, d):
                a.add("op mul uu %s h%d" % (F(Wr[c]), c))
                a.add("op add lt%d lt%d uu" % (r, r))
            a.add("op mul lg%d lt%d %s" % (r, r, F(1.0 / TEMP)))
            a.add("op pow l%d %s lg%d" % (r, E7, r))
        a.add("set ps l%d" % rows[0][0])
        for (r, _) in rows[1:]:
            a.add("op add ps ps l%d" % r)
        a.add("write ps pt%d %d" % (hi // SLOTS, hi % SLOTS))
        # 行粒度落盘：pw 按 token id 索引（V 个槽），采样器据此定出确切行号
        for (r, _) in rows:
            a.add("write l%d pw%d %d" % (r, r // SLOTS, r % SLOTS))
        a.add("write ph hf%d %d" % (hi // SLOTS, hi % SLOTS))
        a.add("set last ph")
        a.jmp("L")
        procs.append(("head%d" % hi, a.render(), []))

    # ---- 采样：三级（head 已把每行 exp 值按 token id 写进 pw 槽）
    # ① gsum：每 GS 个 head 一组，等本组 hf 标志和 >= ph*n 后求本组 pt 和 -> gs0 / gf0
    GS = 128
    ng = (nhead + GS - 1) // GS
    for g in range(ng):
        hi0 = g * GS
        hi1 = min(nhead, hi0 + GS)
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read ph sp0 %d" % SL_GO_B)
        a.jmp("L", "lessThanEq", "ph", "last")
        a.add("op mul tgt ph %d" % (hi1 - hi0))
        a.label("WC")
        a.add("set s 0")
        for hh in range(hi0, hi1):
            a.add("read v hf%d %d" % (hh // SLOTS, hh % SLOTS))
            a.add("op add s s v")
        a.jmp("WC", "lessThan", "s", "tgt")
        a.add("set sm 0")
        for hh in range(hi0, hi1):
            a.add("read v pt%d %d" % (hh // SLOTS, hh % SLOTS))
            a.add("op add sm sm v")
        a.add("write sm gs0 %d" % g)
        a.add("write ph gf0 %d" % g)
        a.add("set last ph")
        a.jmp("L")
        procs.append(("gsum%d" % g, a.render(), []))

    # ② prefix（bar）：等全部 gsum 就绪 -> SL_SCAN_PH = ph（head 已算完本轮的信号）
    a = Asm()
    a.add("setrate 1000")
    a.label("i0")
    a.add("read f sp0 %d" % SL_INIT)
    a.jmp("i0", "equal", "f", "0")
    a.add("set last 0")
    a.label("L")
    a.add("read ph sp0 %d" % SL_GO_B)
    a.jmp("L", "lessThanEq", "ph", "last")
    for g in range(ng):
        a.add("read w gf0 %d" % g)
        a.jmp("L", "lessThan", "w", "ph")
    a.add("write ph sp0 %d" % SL_SCAN_PH)
    a.add("set last ph")
    a.jmp("L")
    procs.append(("prefix", a.render(), []))

    # ③ bsum：pw 按 (cell, SUB 槽) 分块求块和 -> bs0 / bf0
    SUB = 128
    pwchunks = []
    for cell in range((V + SLOTS - 1) // SLOTS):
        cslots = min(SLOTS, V - cell * SLOTS)
        for sub in range((cslots + SUB - 1) // SUB):
            base = sub * SUB
            n = min(SUB, cslots - base)
            pwchunks.append((cell, base, n))
    nchunk = len(pwchunks)
    for ci, (cell, base, n) in enumerate(pwchunks):
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read ph sp0 %d" % SL_GO_B)
        a.jmp("L", "lessThanEq", "ph", "last")
        a.add("read bar sp0 %d" % SL_SCAN_PH)
        a.jmp("L", "lessThan", "bar", "ph")
        a.add("set sm 0")
        a.add("set ad %d" % base)
        a.add("set send %d" % (base + n))
        a.label("B")
        a.add("read pv pw%d ad" % cell)
        a.add("op add sm sm pv")
        a.add("op add ad ad 1")
        a.jmp("B", "lessThan", "ad", "send")
        a.add("write sm bs0 %d" % ci)
        a.add("write ph bf0 %d" % ci)
        a.add("set last ph")
        a.jmp("L")
        procs.append(("bsum%d" % ci, a.render(), []))

    # ④ prefix2：等全部 bsum -> tot / thr / 目标块 tc / 块内前缀和 off
    a = Asm()
    a.add("setrate 1000")
    a.label("i0")
    a.add("read f sp0 %d" % SL_INIT)
    a.jmp("i0", "equal", "f", "0")
    a.add("set last 0")
    a.label("L")
    a.add("read ph sp0 %d" % SL_GO_B)
    a.jmp("L", "lessThanEq", "ph", "last")
    for ci in range(nchunk):
        a.add("read w bf0 %d" % ci)
        a.jmp("L", "lessThan", "w", "ph")
    a.add("set tot 0")
    for ci in range(nchunk):
        a.add("read v bs0 %d" % ci)
        a.add("op add tot tot v")
    # r 表按「本轮序号」索引，而 last 是单调递增的相位 ⇒ 必须 mod N（每轮相位恰好推进 N）
    a.add("op mod ra last %d" % N)
    a.add("op add ra ra %d" % SL_R)
    a.add("read rr sp0 ra")
    a.add("op mul thr rr tot")
    a.add("set cu 0")
    a.add("set tc 0")
    a.add("set off 0")
    a.add("set found 0")
    for ci in range(nchunk):
        a.jmp("SK%d" % ci, "notEqual", "found", "0")
        a.add("read v bs0 %d" % ci)
        a.add("op add cu2 cu v")
        a.jmp("NX%d" % ci, "lessThan", "cu2", "thr")
        a.add("set tc %d" % ci)
        a.add("set off cu")
        a.add("set found 1")
        a.jmp("SK%d" % ci)
        a.label("NX%d" % ci)
        a.add("set cu cu2")
        a.label("SK%d" % ci)
    a.add("write tc sp0 %d" % SL_TARGET)
    a.add("write off sp0 %d" % SL_OFFSET)
    a.add("write thr sp0 %d" % SL_THR)
    a.add("write ph sp0 %d" % SL_THR_PH)
    a.add("set last ph")
    a.jmp("L")
    procs.append(("prefix2", a.render(), []))

    # ⑤ bscan：目标块内从 off 起扫，第一个 cum >= thr 的槽 -> 全局 token 号
    for ci, (cell, base, n) in enumerate(pwchunks):
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read ph sp0 %d" % SL_GO_B)
        a.jmp("L", "lessThanEq", "ph", "last")
        a.add("read tph sp0 %d" % SL_SCAN_PH)
        a.jmp("L", "lessThan", "tph", "ph")
        a.add("read tg sp0 %d" % SL_TARGET)
        a.jmp("L", "notEqual", "tg", str(ci))
        a.add("read cu sp0 %d" % SL_OFFSET)
        a.add("read thr sp0 %d" % SL_THR)
        a.add("set ad %d" % base)
        a.add("set send %d" % (base + n))
        a.label("S")
        a.add("read pv pw%d ad" % cell)
        a.add("op add cu cu pv")
        a.jmp("FD", "greaterThanEq", "cu", "thr")
        a.add("op add ad ad 1")
        a.jmp("S", "lessThan", "ad", "send")
        a.label("FD")
        a.add("op add ix ad %d" % (cell * SLOTS))
        a.add("write ix sp0 %d" % SL_TOKOUT)
        a.add("write ph sp0 %d" % SL_DONE_C)
        a.add("set last ph")
        a.jmp("L")
        procs.append(("bscan%d" % ci, a.render(), []))

    # ---- driver
    a = Asm()
    a.add("setrate 1000")
    a.label("i0")
    a.add("read f sp0 %d" % SL_INIT)
    a.jmp("i0", "equal", "f", "0")
    if cfg.get("input"):
        # 输入模式：等玩家在输入方块里打字（第 0 个字符非 0），再把 S 个 token 装进窗口
        inmsg = "message%d" % (MSG_IN + 1)
        a.add("write 1 sp0 %d" % SL_PHASE)
        a.label("WGO")
        a.label("WG1")
        # 等输入期间心跳：状态面板靠它区分「在等你打字」和「driver 根本没跑」
        a.add("write 1 sp0 %d" % SL_PHASE)
        a.add("read hb sp0 %d" % SL_HB)
        a.add("op add hb hb 1")
        a.add("write hb sp0 %d" % SL_HB)
        a.add("read ic %s 0" % inmsg)
        a.jmp("WG1", "equal", "ic", "0")
        a.add("wait 0.3")          # settle：等分词器把 16 个位置都扫过一遍
        # 相位计数器是**单调递增**的（preload: jmp L lessThanEq g last），清零会永久卡死；
        # 正确做法是接着上一轮的相位往上加：pbase = 上一轮的 GO_B。
        a.add("read gb sp0 %d" % SL_GO_B)
        a.add("set pbase gb")
        a.add("write 0 sp0 %d" % SL_CNT)     # 让 say 处理器知道「新一轮还没出字」
        for j in range(S):
            a.add("read iv sp0 %d" % (SL_IN + j))
            a.add("write iv sp0 %d" % (SL_TOK + j))
    if not cfg.get("input"):
        a.add("set pbase 0")
    a.add("set t 0")
    a.add("write 2 sp0 %d" % SL_PHASE)
    a.label("round")
    a.add("op add p t 1")
    if cfg.get("input"):
        a.add("op add p p pbase")
    a.add("write p sp0 %d" % SL_GO_A)
    for j in range(S):
        a.label("U%d" % j)
        a.add("read pf pf0 %d" % j)
        a.jmp("U%d" % j, "lessThan", "pf", "p")
    a.add("write p sp0 %d" % SL_GO_B)
    a.label("WC")
    a.add("read dc sp0 %d" % SL_DONE_C)
    a.jmp("WC", "lessThan", "dc", "p")
    a.add("read nk sp0 %d" % SL_TOKOUT)
    a.add("op add a t %d" % SL_OUT)
    a.add("write nk sp0 a")
    for sm in range(S - 1):
        a.add("read vv sp0 %d" % (SL_TOK + sm + 1))
        a.add("write vv sp0 %d" % (SL_TOK + sm))
    a.add("write nk sp0 %d" % (SL_TOK + S - 1))
    a.add("op add t t 1")
    a.add("write t sp0 %d" % SL_CNT)
    a.add("write t sp0 %d" % SL_GEN)
    a.jmp("round", "notEqual", "t", str(N))
    if cfg.get("input"):
        # 一轮跑完：等玩家清空输入方块，再回 WGO 等他重新打（边沿触发）
        a.add("write 3 sp0 %d" % SL_PHASE)
        a.label("ARM")
        a.label("AR1")
        a.add("read hb sp0 %d" % SL_HB)
        a.add("op add hb hb 1")
        a.add("write hb sp0 %d" % SL_HB)
        a.add("read ic %s 0" % inmsg)
        a.jmp("AR1", "notEqual", "ic", "0")
        a.jmp("WGO")
    else:
        a.label("halt")
        a.add("wait 0")
        a.jmp("halt")
    procs.append(("driver", a.render(), []))

    # ---- say：把 token 解成真字符打到 message1
    # 每 (输出位置 p, id 段) 一台：各自等 CNT>=N，读 v=SL_OUT+p，命中自己段就 print 一个字。
    # 不握手、不排序 —— 靠「同一 tick 按瓦片顺序执行」让字序 = 处理器放置顺序（p 主序）。
    voc = cfg.get("vocab") or []
    if voc:
        CH = 456
        nch = (V + CH - 1) // CH
        for p in range(N):
            for ci in range(nch):
                lo = ci * CH
                hi = min(V, lo + CH)
                a = Asm()
                a.add("setrate 1000")
                a.label("i0")
                a.add("read f sp0 %d" % SL_INIT)
                a.jmp("i0", "equal", "f", "0")
                a.label("S")
                a.add("read c sp0 %d" % SL_CNT)
                a.jmp("S", "lessThan", "c", str(N))
                a.add("read v sp0 %d" % (SL_OUT + p))
                # 段外必须直接 halt、绝不能 printflush —— printflush 是整块替换，
                # 同一位置的其他 8 台段处理器若也 flush，会把刚打上的字清空（本轮踩过）。
                a.jmp("halt", "lessThan", "v", str(lo))
                a.jmp("halt", "greaterThanEq", "v", str(hi))
                for k in range(lo, hi):
                    ch = voc[k] if k < len(voc) else ""
                    ch = _saychar(ch)
                    # 只跳过本行 print（跳到下一条比较），不能一 mismatch 就跳到末尾——
                    # 否则 v=8 在第 0 条就跳走，字符永远打不出来（本轮踩过）。
                    lab = "k%d" % k
                    a.jmp(lab, "notEqual", "v", str(k))
                    a.add('print "%s"' % ch)
                    a.label(lab)
                a.add("printflush message%d" % (p + 1))
                a.label("halt")
                if cfg.get("input"):
                    # 输入模式要能反复跑：等 driver 把 CNT 清回 <N（新一轮开始）再回到等字
                    a.add("read c2 sp0 %d" % SL_CNT)
                    a.jmp("halt", "greaterThanEq", "c2", str(N))
                    a.jmp("S")
                else:
                    a.add("wait 0")
                    a.jmp("halt")
                procs.append(("say%d_%d" % (p, ci), a.render(), []))

    # ---- tokenizer（游戏内输入）：把玩家打在输入方块里的字映射成 token id
    # 每 (位置 p, id 段 ci) 一台：读 message{N+1} 第 p 个字符码，命中本段就写 SL_IN+p。
    # 段内逐 id 比较（jump 跳 + write，2 条/id）；段外靠 lo/hi 两次比较直接跳过。
    if cfg.get("input") and voc:
        CH = 456
        nch = (V + CH - 1) // CH
        inmsg = "message%d" % (MSG_IN + 1)
        for p in range(S):
            for ci in range(nch):
                lo = ci * CH
                hi = min(V, lo + CH)
                codes = []
                for k in range(lo, hi):
                    ch = voc[k] if k < len(voc) else ""
                    if isinstance(ch, str) and len(ch) == 1:
                        codes.append((k, ord(ch)))
                a = Asm()
                a.add("setrate 1000")
                a.label("i0")
                a.add("read f sp0 %d" % SL_INIT)
                a.jmp("i0", "equal", "f", "0")
                a.label("L")
                a.add("read c %s %d" % (inmsg, p))
                if ci == 0:
                    # 空位（字符码 0）→ id 0，与 id2ch[0] 一致
                    a.jmp("NZ", "notEqual", "c", "0")
                    a.add("write 0 sp0 %d" % (SL_IN + p))
                    a.add("wait 0.1")
                    a.jmp("L")
                    a.label("NZ")
                if codes:
                    a.jmp("NX", "lessThan", "c", str(min(c for _, c in codes)))
                    a.jmp("NX", "greaterThan", "c", str(max(c for _, c in codes)))
                    for k, c in codes:
                        lab = "k%d" % k
                        a.jmp(lab, "notEqual", "c", str(c))
                        a.add("write %d sp0 %d" % (k, SL_IN + p))
                        a.label(lab)
                a.label("NX")
                a.add("wait 0.1")
                a.jmp("L")
                procs.append(("tok%d_%d" % (p, ci), a.render(), []))

    # ---- printer（打到 message1，打 token 数字；给了 --vocab 时被 say 取代）
    if not voc:
        a = Asm()
        a.add("setrate 1000")
        a.label("i0")
        a.add("read f sp0 %d" % SL_INIT)
        a.jmp("i0", "equal", "f", "0")
        a.add("set last 0")
        a.label("L")
        a.add("read c sp0 %d" % SL_CNT)
        a.jmp("L", "lessThanEq", "c", "last")
        a.add("set i 0")
        a.label("P")
        a.jmp("PD", "greaterThanEq", "i", "c")
        a.add("op add a i %d" % SL_OUT)
        a.add("read vv sp0 a")
        a.add("print vv")
        # 每 4 个 token 分组：message 方块在第一处换行处截断（实测），所以用双空格
        a.add("op mod lm i 4")
        a.jmp("SP", "notEqual", "lm", "3")
        a.add('print "  "')
        a.jmp("NX")
        a.label("SP")
        a.add('print " "')
        a.label("NX")
        a.add("op add i i 1")
        a.jmp("P")
        a.label("PD")
        a.add("printflush message1")
        a.add("set last c")
        a.jmp("L")
        procs.append(("printer", a.render(), []))

    # ---- reporter：游戏内状态面板（说明方块 + 实时阶段）。只在 say 模式（有 --vocab）时加
    if LOG_PANEL:
        a = Asm()
        a.add("setrate 1000")
        a.add('print "[用法] 右键点最右边的方块, 输入或粘贴中文 -> 等约60秒 -> 左边方块出字 -> 清空输入框再打下一句"')
        a.add("printflush message%d" % (MSG_INSTR + 1))
        a.add("set hp -1")
        a.label("L")
        a.add("read ph sp0 %d" % SL_PHASE)
        a.jmp("BOOT", "lessThan", "ph", "1")
        a.jmp("L2", "equal", "ph", "2")
        a.jmp("L3", "equal", "ph", "3")
        # ph == 1：等输入。心跳不动 => driver 没在跑（最常见：没进 playtest / 蓝图链接断）
        a.add("read hb sp0 %d" % SL_HB)
        a.jmp("ALIVE", "notEqual", "hb", "hp")
        a.add('print "[静止] 处理器没在跑: 请在编辑器里按播放(playtest), 并检查蓝图链接"')
        a.jmp("FL")
        a.label("ALIVE")
        a.add("set hp hb")
        a.add('print "[就绪] 在输入方块里打字, 8 字约 60 秒"')
        a.jmp("FL")
        a.label("L2")
        a.add("read gn sp0 %d" % SL_GEN)
        a.add('print "[生成] "')
        a.add("print gn")
        a.add('print " / %d 字"' % N)
        a.jmp("FL")
        a.label("L3")
        a.add('print "[完成] 清空输入方块, 再打下一句"')
        a.jmp("FL")
        a.label("BOOT")
        a.add("read pg sp0 %d" % SL_PROG)
        a.add("read pn sp0 %d" % SL_PROGN)
        a.jmp("BOOT2", "greaterThan", "pn", "0")
        a.add('print "[启动] 初始化中..."')
        a.jmp("FL")
        a.label("BOOT2")
        a.add('print "[启动] 载入嵌入 "')
        a.add("print pg")
        a.add('print " / "')
        a.add("print pn")
        a.label("FL")
        a.add("printflush message%d" % (MSG_STAT + 1))
        a.add("wait 0.4")
        a.jmp("L")
        procs.append(("reporter", a.render(), []))

    # ---- 布局
    cellpos = {}
    cc = int(math.ceil(math.sqrt(len(allcells))))
    for i, nm in enumerate(allcells):
        cellpos[nm] = (i % cc, i // cc)
    ncellrows = (len(allcells) + cc - 1) // cc
    # say 模式：每个输出位置一个 message 方块（printflush 是「整块替换」，多台抢同一块会互相清空）
    # 其后依次是说明方块、状态方块、（输入模式）输入方块
    nmsg = (nout + 2 + (1 if cfg.get("input") else 0)) if voc else 1
    msgpos = [(cc + 2 + 2 * k, 0) for k in range(nmsg)]
    y0 = ncellrows + 2
    pw = int(math.ceil(math.sqrt(len(procs))))
    procpos = {}
    for i, (nm, _, _) in enumerate(procs):
        procpos[nm] = (i % pw, y0 + i // pw)

    def make_links(nm, lines, first):
        px, py = procpos[nm]
        seen = set(first)
        used = []
        for ln in lines:
            t = ln.split()
            if len(t) >= 4 and t[0] in ("read", "write") and t[2] in cellpos:
                if t[2] not in seen:
                    seen.add(t[2])
                    used.append(t[2])
        out = [Link(x, cellpos[x][0] - px, cellpos[x][1] - py) for x in first]
        for x in sorted(used):
            out.append(Link(x, cellpos[x][0] - px, cellpos[x][1] - py))
        if nm == "printer":
            out.append(Link("message1", msgpos[0][0] - px, msgpos[0][1] - py))
        elif nm.startswith("say"):
            _p = int(nm[3:].split("_")[0])
            out.append(Link("message%d" % (_p + 1),
                            msgpos[_p][0] - px, msgpos[_p][1] - py))
        elif nm.startswith("tok") or (nm == "driver" and cfg.get("input")):
            # 输入方块 = 最后一个 message
            _q = nmsg - 1
            out.append(Link("message%d" % nmsg, msgpos[_q][0] - px, msgpos[_q][1] - py))
        if nm == "reporter":
            # 说明方块 + 状态方块（printflush 替换整块，所以要各占一块）
            for _mi in (MSG_INSTR, MSG_STAT):
                out.append(Link("message%d" % (_mi + 1),
                                msgpos[_mi][0] - px, msgpos[_mi][1] - py))
        return out

    if verbose:
        print("W\tprocs %d cells %d\t%.1fs" % (len(procs), len(allcells) + 1, time.time() - t0), flush=True)

    bp = Blueprint("closure_wpl_d%ds%dl%dv%d" % (d, S, L, V), "", [])
    for nm in allcells:
        x, y = cellpos[nm]
        bp.add_block("world-cell", x, y)
    for k in range(nmsg):
        bp.add_block("message", msgpos[k][0], msgpos[k][1])
    for (nm, lines, first) in procs:
        x, y = procpos[nm]
        bp.add_logic("\n".join(lines), x, y, block="world-processor",
                     links=make_links(nm, lines, first))
    mb = bp.build()
    with open(path, "wb") as f:
        f.write(mb)
    build_s = time.time() - t0

    # ---- 参考序列
    ref, zinfo = sample_ref(W, cfg)

    maxlines = max(len(x[1]) for x in procs)
    over = [x[0] for x in procs if len(x[1]) > LIMIT]
    stats = dict(
        d=d, S=S, L=L, V=V, N=N, cap=cap, seed=cfg["seed"], seed_tok=cfg["seed_tok"],
        TEMP=TEMP, head_rows=hr,
        units=len(units), nfwd=nfwd, nhead=nhead, nloader=nloader, nlv1=nlv1,
        nsplit=nsplit, nprocs=len(procs), ncells=len(allcells) + 1,
        nworldcell=len(allcells), nemb=nemb, nval=len(order), nvalcell=nvalcell,
        nacell=nacell, nsplits_rounds=nsplit,
        total_stmts=sum(len(x[1]) for x in procs),
        fwd_stmts=sum(lens), maxlines=maxlines, over=over,
        build_s=round(build_s, 1), bytes=len(mb),
        ref_tokens=ref, zinfo=zinfo,
        input=bool(cfg.get("input")), input_idx=(nmsg - 1) if cfg.get("input") else None,
        msg_instr=(MSG_INSTR + 1) if voc else None, msg_stat=(MSG_STAT + 1) if voc else None,
        msg_in=(MSG_IN + 1) if MSG_IN is not None else None,
        prompt=list(cfg.get("prompt") or []),
        cellpos={k: list(v) for k, v in cellpos.items()},
        msgpos=[list(v) for v in msgpos], procpos={k: list(v) for k, v in procpos.items()},
        layout=dict(cc=cc, ncellrows=ncellrows, y0=y0, pw=pw),
    )
    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    if verbose:
        print("W\tbuilt %s  %.2f MB  %d procs  %d cells  max %d lines  %.1fs"
              % (os.path.basename(path), len(mb) / 1e6, len(procs), len(allcells) + 1,
                 maxlines, build_s), flush=True)
        if over:
            print("W\t!! OVER LIMIT: %s" % over[:5], flush=True)
        print("W\tref tokens %s" % ref, flush=True)
    return stats


# ---------------------------------------------------------------- 运行（分批装载 + 分片 tick）
PLACE_JS = r"""
var cs = Vars.content;
var tm = Packages.mindustry.game.Team.sharded;
var T = %s;
for (var i = 0; i < T.length; i++) {
  Vars.world.tile(T[i].x, T[i].y).setBlock(cs.block(T[i].block), tm);
}
"""

CONF_JS = r"""
var LLX = Packages.mindustry.world.blocks.logic.LogicBlock.LogicLink;
var T = %s;
for (var i = 0; i < T.length; i++) {
  var b = Vars.world.build(T[i].x, T[i].y);
  b.code = T[i].logic;
  b.links.clear();
  for (var j = 0; j < T[i].links.length; j++) {
    var l = T[i].links[j];
    b.links.add(new LLX(l.x, l.y, l.name, true));
  }
  b.updateCode(b.code);
  if (!b.executor.initialized()) throw "PREFLIGHT " + T[i].x + "," + T[i].y;
  for (var k = 0; k < b.links.length; k++) {
    if (!b.links.get(k).valid) throw "LINK " + T[i].x + "," + T[i].y + " " + k;
  }
  b.executor.team = Packages.mindustry.game.Team.sharded;
  b.efficiency = 1;
  procs.push(b);
}
print("CONF " + procs.length);
"""

PROBE_JS = r"""
var __f = function (o, n) {
  var fs = o.getClass().getDeclaredFields();
  for (var i = 0; i < fs.length; i++) {
    if (String(fs[i].getName()) === n) { fs[i].setAccessible(true); return fs[i].get(o); }
  }
  return null;
};
var sp = Vars.world.build(%d, %d);
var mem = __f(sp, "numberMemory");
var arr = [];
for (var i = 0; i < 512; i++) arr.push(String(mem[i]));
print("SP0 " + arr.join(","));
var MGS = %s;
for (var q_ = 0; q_ < MGS.length; q_++) {
  print("MSG" + q_ + "|" + String(__f(Vars.world.build(MGS[q_][0], MGS[q_][1]), "message")));
}
var mg = MGS.length ? Vars.world.build(MGS[0][0], MGS[0][1]) : null;
if (%d) {
  var fs2 = mg.getClass().getDeclaredFields();
  for (var q = 0; q < fs2.length; q++) {
    fs2[q].setAccessible(true);
    print("MF " + fs2[q].getName() + " = " + String(fs2[q].get(mg)));
  }
}
if (%d >= 2) {
  for (var p = 0; p < procs.length; p++) {
    if (String(procs[p].code).indexOf("printflush") >= 0) {
      var vs = procs[p].executor.vars;
      var s2 = [];
      for (var v = 0; v < vs.length; v++) {
        var q2 = vs[v];
        s2.push(String(q2.name) + "=" + (q2.isobj ? String(q2.objval) : String(q2.numval)));
      }
      print("PV " + s2.join(" "));
    }
  }
}
var MKS = %s;
for (var m2 = 0; m2 < MKS.length; m2++) {
  var MK = MKS[m2];
  for (var p2 = 0; p2 < procs.length; p2++) {
    var c2 = String(procs[p2].code);
    if (c2.indexOf(MK) >= 0) {
      var vs2 = procs[p2].executor.vars;
      var s3 = [];
      for (var v2 = 0; v2 < vs2.length; v2++) {
        var q3 = vs2[v2];
        s3.push(String(q3.name) + "=" + (q3.isobj ? String(q3.objval) : String(q3.numval)));
      }
      print("DV[" + MK + "] " + s3.join(" "));
    }
  }
}
print("NPROC " + procs.length);
"""

SETMSG_JS = r"""
var __f = function (o, n) {
  var fs = o.getClass().getDeclaredFields();
  for (var i = 0; i < fs.length; i++) {
    if (String(fs[i].getName()) === n) { fs[i].setAccessible(true); return fs[i].get(o); }
  }
  return null;
};
var mb = Vars.world.build(%d, %d);
var sb = __f(mb, "message");
sb.setLength(0);
var TXT = %s;
if (TXT.length > 0) { sb.append(TXT); }
print("SETMSG " + String(__f(mb, "message")));
"""

PT_JS = r"""
var __g = function (o, n) {
  var fs = o.getClass().getDeclaredFields();
  for (var i = 0; i < fs.length; i++) {
    if (String(fs[i].getName()) === n) { fs[i].setAccessible(true); return fs[i].get(o); }
  }
  return null;
};
var CELLS = %s;
var CNT = %s;
for (var k = 0; k < CELLS.length; k++) {
  var mem = __g(Vars.world.build(CELLS[k][0], CELLS[k][1]), "numberMemory");
  var a = [];
  for (var i = 0; i < CNT[k]; i++) a.push(String(mem[i]));
  print("PT" + k + " " + a.join(","));
}
"""


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run(args):
    """把构建好的 .msch 丢进 Mindustry 跑闭环（需要 jar + 姊妹仓的 mlogsim）。"""
    try:
        import mlogsim as M              # 姊妹仓 mlog-agentskill 的 tools/mlogsim.py
    except ImportError:
        raise SystemExit(
            "运行侧需要 Mindustry 模拟器：\n"
            "  1) 装 Mindustry v160.7 server jar + JRE21\n"
            "  2) 从 https://github.com/EnernalIliary/mlog-agentskill 取 tools/mlogsim.py\n"
            "  3) 让它能在 PYTHONPATH 里被 import（或把那份 tools/ 加进 PYTHONPATH）\n"
            "只想构建蓝图的话不需要这些，去掉 --run 即可。")
    try:
        from . import msch as MS
    except ImportError:
        import msch as MS

    path = args.run
    sch = MS.parse_file(path)
    if not sch.tiles:
        raise SystemExit("empty msch: %s" % path)
    minx = min(t.x for t in sch.tiles)
    miny = min(t.y for t in sch.tiles)
    base = args.base
    tx, ty = base - minx, base - miny
    tiles = []
    for t in sch.tiles:
        e = {"block": t.block, "x": t.x + tx, "y": t.y + ty,
             "logic": (t.logic.code if t.logic is not None else None), "links": []}
        if t.logic is not None:
            e["links"] = [{"name": l.name, "x": l.x + tx, "y": l.y + ty}
                          for l in t.logic.links]
        tiles.append(e)
    nblk = len(tiles)
    coded = [t for t in tiles if t.get("logic")]
    print("R\ttiles %d (logic %d) shift (%d,%d)" % (nblk, len(coded), tx, ty), flush=True)

    st_path = args.stats or (path[:-5] + ".json" if path.endswith(".msch") else None)
    stats = None
    if st_path and os.path.exists(st_path):
        with open(st_path, encoding="utf-8") as f:
            stats = json.load(f)
    if stats is None:
        raise SystemExit("need stats json (build first, or pass --stats)")
    # sp0/message 世界坐标
    sx, sy = stats["cellpos"]["sp0"]
    mp = stats["msgpos"]
    if mp and isinstance(mp[0], (int, float)):   # 兼容旧的单 message 统计格式
        mp = [mp]
    mgs = [[v[0] + tx, v[1] + ty] for v in mp]
    sx, sy = sx + tx, sy + ty

    allx = [t["x"] for t in tiles]
    ally = [t["y"] for t in tiles]
    ext = max(max(allx), max(ally)) + 4
    world = args.world or max(64, int(math.ceil((ext + 8) / 10.0)) * 10)
    print("R\tworld %d  span %d..%d" % (world, min(allx), max(allx)), flush=True)

    t0 = time.time()
    srv = M.Server(M._default_java(), M._default_jar(), world, None)
    srv.js_timeout = args.js_timeout
    try:
        srv.prepare()
    except Exception as e:      # pragma: no cover
        print("R\tprepare: %s" % e, flush=True)
    print("R\tprepared %.1fs" % (time.time() - t0), flush=True)
    # 关掉自动结束/换图：黑世界没有核心，attack 判定会在 ~40 s 后判我方败并换图，
    # 换图会把整个方块世界清空（Vars.world.build() 变 null），闭环必然半路死掉。
    try:
        srv.js("Vars.state.rules.canGameOver = false; Vars.state.rules.attackMode = false;"
               " Vars.state.rules.waves = false;"
               " print('RULES ' + String(Vars.state.rules.canGameOver));")
    except Exception as e:      # pragma: no cover
        print("R\trules: %s" % e, flush=True)

    # PLACE
    n = 0
    for bt in _batches(tiles, 480):
        js = PLACE_JS % json.dumps([{"block": t["block"], "x": t["x"], "y": t["y"]} for t in bt])
        srv.js(js)
        n += len(bt)
        if n % 9600 < 480:
            print("R\tplace %d/%d %.1fs" % (n, nblk, time.time() - t0), flush=True)
    print("R\tplaced %d %.1fs" % (nblk, time.time() - t0), flush=True)

    # CONF
    srv.js("var procs = [];")
    n = 0
    for bt in _batches(coded, 40):
        js = CONF_JS % json.dumps([{"x": t["x"], "y": t["y"], "logic": t["logic"],
                                    "links": t.get("links") or []} for t in bt])
        out = srv.js(js)
        for ln in out:
            if "PREFLIGHT" in ln or "LINK " in ln:
                raise SystemExit("conf fail: %s" % ln)
        n += len(bt)
        if n % 2000 < 40:
            print("R\tconf %d/%d %.1fs" % (n, len(coded), time.time() - t0), flush=True)
    print("R\tconf %d %.1fs" % (len(coded), time.time() - t0), flush=True)

    # 输入模式：把「玩家打的字」写进输入 message 方块（世界外注入，等价于玩家在方块里编辑）
    if args.input_text is not None and stats.get("input_idx") is not None:
        _q = int(stats["input_idx"])
        try:
            out = srv.js(SETMSG_JS % (mgs[_q][0], mgs[_q][1], json.dumps(args.input_text)))
            print("R\tinput %r -> %s" % (args.input_text, (out or [""])[-1]), flush=True)
        except Exception as e:
            print("R\tinput inject FAIL: %s" % e, flush=True)


    N = stats["N"]
    S = stats.get("S", 16)
    dump = PROBE_JS % (sx, sy, json.dumps(mgs), args.msgfields, args.msgfields,
                       json.dumps([x for x in args.dumpvars.split(",") if x]))
    nslices = args.slice
    total = 0
    hist = []
    done_at = None
    done2_at = None
    _in_q = int(stats["input_idx"]) if stats.get("input_idx") is not None else None
    _sent2 = 0
    _re_armed = False
    res2 = {}
    while True:
        loop = M._scene_loop_js(nslices, True, "all", 1.0, False, False, "logic", False)
        out = srv.js("var procs = procs; " + loop + dump)
        total += nslices
        if not any(ln.startswith("SP0 ") for ln in out):
            print("R\tWORLD RESET or dump fail at tick %d; tail=%s" % (total, out[-3:]), flush=True)
            break
        sp0 = None
        msgs = {}
        for ln in out:
            if ln.startswith("SP0 "):
                sp0 = [x for x in ln[4:].split(",")]
                sp0 = sp0 + ["?"] * (512 - len(sp0))
            elif ln.startswith("MSG") and "|" in ln:
                k_, _, txt = ln[3:].partition("|")
                try:
                    msgs[int(k_)] = txt
                except Exception:
                    pass
            elif (ln.startswith("MF ") or ln.startswith("CONF") or ln.startswith("NPROC")
                  or ln.startswith("PV ") or ln.startswith("DV")):
                print("R\t  " + ln, flush=True)
        msg = "".join(msgs.get(q, "") for q in range(len(mgs)))
        if sp0 is None:
            print("R\tno dump at tick %d; out=%s" % (total, out[-3:]), flush=True)
            break
        vals = {}
        for i, v in enumerate(sp0):
            vals[i] = v
        cnt = sp0[SL_CNT]
        toks = [sp0[SL_OUT + t] for t in range(N)]
        print("R\ttick %d  cnt %s  out %s  go %s/%s done %s thrph %s scanph %s"
              "  thr %s tg %s off0 %s  in %s  msg %r"
              % (total, cnt, toks, sp0[SL_GO_A], sp0[SL_GO_B], sp0[SL_DONE_C],
                 sp0[SL_THR_PH], sp0[SL_SCAN_PH], sp0[SL_THR], sp0[SL_TARGET],
                 sp0[SL_OFFSET], sp0[SL_IN:SL_IN + S], msg), flush=True)
        hist.append(dict(ticks=total, cnt=cnt, tokens=toks, msg=msg,
                         go_a=sp0[SL_GO_A], go_b=sp0[SL_GO_B], done_c=sp0[SL_DONE_C],
                         thr_ph=sp0[SL_THR_PH]))
        # 第二轮：清空输入方块（ARM 需要看到 0）→ 隔一次再打新字 → 等 cnt 掉下来再涨回去
        if args.input_text2 is not None and _in_q is not None and done_at is not None:
            if _sent2 == 0:
                srv.js(SETMSG_JS % (mgs[_in_q][0], mgs[_in_q][1], json.dumps("")))
                _sent2 = 1
            elif _sent2 == 1:
                srv.js(SETMSG_JS % (mgs[_in_q][0], mgs[_in_q][1], json.dumps(args.input_text2)))
                _sent2 = 2
                print("R\tinput2 %r -> 已注入" % args.input_text2, flush=True)
            elif _sent2 == 2:
                try:
                    _c2 = int(float(cnt))
                except Exception:
                    _c2 = N
                if _c2 < N:
                    _re_armed = True
                elif _re_armed and done2_at is None:
                    done2_at = total
                    res2 = dict(tokens=toks, msg=msg)
                    print("R\tSECOND RUN done at tick %d  out %s  msg %r"
                          % (total, toks, msg), flush=True)
        try:
            done = int(float(cnt)) >= N
        except Exception:
            done = False
        if done and done_at is None:
            done_at = total
        # 完成后再多跑 --extra 个 tick：say 处理器要等 CNT>=N 才落字
        if args.input_text2 is not None:
            _fin = (done2_at is not None and total >= done2_at + args.extra)
        else:
            _fin = (done_at is not None and total >= done_at + args.extra)
        if _fin or total >= args.max_ticks:
            break
    print("R\tdone at %d ticks (%.1fs)" % (total, time.time() - t0), flush=True)
    ptvals = None
    if args.ptcheck and total:
        cp = stats["cellpos"]
        nhead = stats["nhead"]
        cells, cnts = [], []
        k = 0
        while ("pt%d" % k) in cp:
            cells.append([cp["pt%d" % k][0] + tx, cp["pt%d" % k][1] + ty])
            cnts.append(min(SLOTS, nhead - k * SLOTS))
            k += 1
        out = srv.js(PT_JS % (json.dumps(cells), json.dumps(cnts)))
        ptvals = []
        for ln in out:
            if ln.startswith("PT"):
                body = ln.split(" ", 1)[1]
                ptvals.append([float(x) for x in body.split(",") if x])
        flat = [v for c in ptvals for v in c]
        print("R\tptcheck(phase %s) %d cells %d values  sum %.12g  max %.12g  min %.12g"
              % (hist[-1]["go_b"] if hist else "?", len(ptvals), len(flat),
                 sum(flat), max(flat), min(flat)), flush=True)
    res = dict(path=path, ticks=total, hist=hist, nblk=nblk, ncoded=len(coded),
               pt=ptvals,
               world=world, wall_s=round(time.time() - t0, 1),
               ref=stats.get("ref_tokens"))
    if hist:
        res["tokens"] = hist[-1]["tokens"]
        res["msg"] = hist[-1]["msg"]
    if res2:
        res.update(tokens2=res2.get("tokens"), msg2=res2.get("msg"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    try:
        srv.close()
    except Exception:
        pass
    return res


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        prog="gen.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="把一个小 transformer 展开成 Mindustry mlog 蓝图；"
                    "带 --run 时把蓝图丢进游戏里跑闭环。")
    ap.add_argument("--build", action="store_true",
                    help="构建蓝图（这就是默认行为，写上让命令自解释）")
    ap.add_argument("--d", type=int, default=128, help="模型宽度")
    ap.add_argument("--S", type=int, default=16, help="窗口长度")
    ap.add_argument("--L", type=int, default=4, help="层数")
    ap.add_argument("--V", type=int, default=4096, help="词表大小")
    ap.add_argument("--N", type=int, default=4, help="生成多少个 token")
    ap.add_argument("--cap", type=int, default=900, help="每台处理器的行数上限")
    ap.add_argument("--seed", type=int, default=7, help="没有 --weights 时随机权重的种子")
    ap.add_argument("--seed-tok", type=int, default=0, help="无条件种子 token")
    ap.add_argument("--head-rows", type=int, default=3, help="每台 head 处理器算几行 logits")
    ap.add_argument("--temp", type=float, default=1.0, help="采样温度")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="训练好的 fp32 npz（默认仓库自带权重；不存在则用随机权重）")
    ap.add_argument("--out", default=None,
                    help="输出 .msch（默认 ./mlogpt_d<cfg>.msch）")
    ap.add_argument("--stats", default=None, help="stats json（默认与 .msch 同名）")
    ap.add_argument("--run", default=None,
                    help="不构建，改为运行现成 .msch（需 Mindustry jar + 模拟器）")
    ap.add_argument("--slice", type=int, default=20, help="--run 时每轮打印多少个 tick")
    ap.add_argument("--ticks", type=int, default=0,
                    help="--run 总共跑多少个 tick（覆盖 --max-ticks）")
    ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--extra", type=int, default=8,
                    help="token 出齐后再多跑几个 tick（say 处理器此时才落字）")
    ap.add_argument("--base", type=int, default=2, help="蓝图平移到世界坐标的基准")
    ap.add_argument("--world", type=int, default=0, help="世界尺寸（0 = 按蓝图跨度自动）")
    ap.add_argument("--js-timeout", type=int, default=300)
    ap.add_argument("--json", default=None, help="--run 时把结果写到这个 json")
    ap.add_argument("--msgfields", type=int, default=0)
    ap.add_argument("--dumpvars", default="",
                    help="逗号分隔的标记串；代码含该串的处理器全部打印变量")
    ap.add_argument("--ptcheck", action="store_true",
                    help="dump head 局部 exp 和并以 json 落盘")
    ap.add_argument("--vocab", default=DEFAULT_VOCAB,
                    help="词表 json；给了就用 say 处理器把 token id 解成真字符打到 message1")
    ap.add_argument("--table-free", action="store_true",
                    help="无表嵌入：X 在游戏内用 sin 现算，不建嵌入 cell，也不要 weights 里的 E")
    ap.add_argument("--input", action="store_true",
                    help="游戏内输入模式：多 1 个 message 方块当输入，分词器把玩家打的字映射成 token")
    ap.add_argument("--input-text2", default=None,
                    help="运行侧：第一轮跑完后再注入第二段文本，验证边沿触发能重跑")
    ap.add_argument("--input-text", default=None,
                    help="运行侧：把这段文本注入输入方块（构建侧据此算 numpy 参考）")
    a = ap.parse_args()
    if a.ticks:
        a.max_ticks = a.ticks
    if a.run:
        run(a)
        return
    cfg = dict(d=a.d, S=a.S, L=a.L, V=a.V, N=a.N, cap=a.cap, seed=a.seed,
               seed_tok=a.seed_tok, head_rows=a.head_rows, TEMP=a.temp,
               table_free=bool(a.table_free), input=bool(a.input))
    if a.vocab and os.path.exists(a.vocab):
        with open(a.vocab, "r", encoding="utf-8") as f:
            vo = json.load(f)
        cfg["vocab"] = vo["id2ch"] if isinstance(vo, dict) else vo
    elif a.vocab:
        print("gen: 词表 %s 不存在，跳过解码（不影响生成的 mlog）" % a.vocab,
              file=sys.stderr)
    if a.input and cfg.get("vocab"):
        ids = text_to_ids(a.input_text or "", cfg["vocab"])[:a.S]
        cfg["prompt"] = ids + [0] * (a.S - len(ids))
    if a.weights and os.path.exists(a.weights):
        # 权重和模型尺寸对不上时给一句人话，而不是让 numpy 抛 matmul 形状错误
        sh = peek_shapes(a.weights)
        d_w, V_w = sh["d"], sh["V"]
        if (d_w, V_w) != (a.d, a.V):
            raise SystemExit(
                "权重与模型尺寸不匹配：%s 是 d=%d V=%d，而命令里是 d=%d V=%d。\n"
                "  全尺寸: --d 128 --S 16 --L 4 --V 4096 --weights weights/mlogpt-zh-0.72m-fp32.npz\n"
                "  玩具  : --d 8  --S 8  --L 2 --V 32   --weights weights/toy-d8s8l2v32.npz"
                % (a.weights, d_w, V_w, a.d, a.V))
        W = load_weights(cfg, a.weights)
    elif a.weights and a.weights != DEFAULT_WEIGHTS:
        raise SystemExit("weights not found: %s" % a.weights)
    else:
        print("gen: 没找到 %s，改用随机权重（--seed %d）" % (a.weights, a.seed),
              file=sys.stderr)
        W = gen_weights(cfg)
    out = a.out or os.path.join(".", "mlogpt_d%ds%dl%dv%d.msch"
                                % (a.d, a.S, a.L, a.V))
    odir = os.path.dirname(os.path.abspath(out))
    if odir:
        os.makedirs(odir, exist_ok=True)
    st = a.stats or ((out[:-5] if out.endswith(".msch") else out) + ".json")
    build(cfg, W, out, st)


if __name__ == "__main__":
    main()
