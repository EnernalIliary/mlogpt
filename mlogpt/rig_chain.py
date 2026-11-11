# -*- coding: utf-8 -*-
"""rig_chain.py — 在 rig.py 之上做「关键路径压缩」。

改动只有一处语义：把 rig 里「一个 unit 内部串行 d 次 mul+add」的
累加/归约，拆成「叶组 unit + 树形归约 unit」，让关键路径（critpath.py 的
cost = len(stmts)+len(uses)+len(usec) 逐 unit 串行）从 O(d) 降到
O(leaf + log(d/leaf))。

必须拆成独立 Unit 的原因：critpath 把每个 unit 的语句全部当串行，
只在 unit 之间并行；所以树形归约的每个内部节点都必须是自己的 Unit。

数值上加法顺序变了（float64 末位会差 ~1e-16），只影响末位。
"""
import math

try:                                  # 作为包导入：python mlogpt/gen.py
    from . import rig as base
except ImportError:                   # 直接运行本文件时
    import rig as base

F, vkey, Unit = base.F, base.vkey, base.Unit

TREE_W = 8    # 每个归约 unit 最多做 8 次 add
MV_LEAF = 16  # matvec 一行的叶组列数
GROUP = 8     # 元素级语句（h/n/p/exp/c2）每 unit 的条数


# ------------------------------------------------------- 归约
def reduce_keys(units, li, keys, tag, op="add", w=TREE_W):
    """把若干已产出的 key 用树形归约合成一个 key；返回最终 key。"""
    cur = list(keys)
    lvl = 0
    while len(cur) > 1:
        nxt, i = [], 0
        while i < len(cur):
            grp = cur[i:i + w]
            if len(grp) == 1:
                nxt.append(grp[0])
                i += 1
                continue
            u = Unit("%s_L%d_%d" % (tag, lvl, i))
            k = (li, tag, "L", lvl, i)
            acc = u.produce(k)
            u.add("op %s %s %s %s" % (op, acc, u.use(grp[0]), u.use(grp[1])))
            for c in grp[2:]:
                u.add("op %s %s %s %s" % (op, acc, acc, u.use(c)))
            nxt.append(k)
            units.append(u)
            i += w
        cur = nxt
        lvl += 1
    return cur[0]


def _final(units, name, key, src_key, stmts):
    """收尾 unit：读 src_key，写 key，附加语句（stmts 是字符串模板列表，
    可用 {acc} / {src} / {key}）。"""
    u = Unit(name)
    src = u.use(src_key)
    acc = u.produce(key)
    kv = "s_" + vkey(key)
    for st in stmts:
        u.add(st.format(acc=acc, src=src, key=kv))
    units.append(u)
    return u


def _tree_dot(units, li, tag, terms, leaf=MV_LEAF):
    """terms: 元素列表，每个是 (a_kind, a_val, b_kind, b_val)。
    kind='lit' → 立即数；'key' → 读 cell 值；'const' → key 带常量值。
    返回叶组 key 列表（已树形累加过组内）。"""
    keys = []
    for s0 in range(0, len(terms), leaf):
        grp = terms[s0:s0 + leaf]
        u = Unit("%s_lf%d" % (tag, s0))
        prods = []
        for (ak, av, bk, bv) in grp:
            if ak == "lit":
                ia = F(av)
            elif ak == "const":
                ia = u.use_const(av[0], av[1])
            else:
                ia = u.use(av)
            if bk == "lit":
                ib = F(bv)
            elif bk == "const":
                ib = u.use_const(bv[0], bv[1])
            else:
                ib = u.use(bv)
            t = u.tmp("m%d" % len(prods))
            u.add("op mul %s %s %s" % (t, ia, ib))
            prods.append(t)
        cur, lvl = prods, 0
        while len(cur) > 1:          # 组内树形（宽度 TREE_W）
            nxt, i = [], 0
            while i < len(cur):
                sub = cur[i:i + TREE_W]
                if len(sub) == 1:
                    nxt.append(sub[0])
                    i += 1
                    continue
                t = u.tmp("a%d_%d" % (lvl, i))
                u.add("op add %s %s %s" % (t, sub[0], sub[1]))
                for c in sub[2:]:
                    u.add("op add %s %s %s" % (t, t, c))
                nxt.append(t)
                i += TREE_W
            cur = nxt
            lvl += 1
        k = (li, tag, "lf", s0)
        u.add("set %s %s" % (u.produce(k), cur[0]))
        keys.append(k)
        units.append(u)
    return keys


def _mv_row(units, li, name, Mrow, src, post=None):
    """一行 matvec：Σ_c M[r][c]*x_c，树形；src(c) → term 描述。"""
    terms = []
    for c in range(len(Mrow)):
        s = src(c)
        if s[0] == "const":
            terms.append(("lit", Mrow[c], "const", (s[1], s[2])))
        else:
            terms.append(("lit", Mrow[c], "key", s[1]))
    keys = _tree_dot(units, li, "%s_r%d" % (name, r_target[0]), terms)
    if len(keys) > 1:   # 多叶组：先树形归约到单值
        keys = [reduce_keys(units, li, keys, "%s_r%d_red" % (name, r_target[0]))]
    final_key = (li, name, r_target[0])
    stmts = ["set {acc} {src}"]
    if post:
        stmts += post
    _final(units, "%s_r%d_fin" % (name, r_target[0]), final_key, keys[0], stmts)
    return final_key


# 上面 _mv_row 用全局暂存 r（避免再塞参数），单线程生成够用
r_target = [0]


def mv_rows(units, li, name, M, src, post=None):
    for r in range(len(M)):
        r_target[0] = r
        _mv_row(units, li, name, M[r], src, post)
    return units


def dot_rows(units, li, name, n, terms_fn, post=None):
    for r in range(n):
        terms = terms_fn(r)
        keys = _tree_dot(units, li, "%s_r%d" % (name, r), terms)
        if len(keys) > 1:
            keys = [reduce_keys(units, li, keys, "%s_r%d_red" % (name, r))]
        stmts = ["set {acc} {src}"] + (post or [])
        _final(units, "%s_r%d_fin" % (name, r), (li, name, r), keys[0], stmts)
    return units


# ------------------------------------------------------- 一层
def layer_units_chain(cfg, li, W, g, b, X):
    d, S = cfg["d"], cfg["S"]
    units = []
    sqrt_d = math.sqrt(d)

    def mk_src(j):
        """第 j 个上下文的输入源：li==0 或 j<S-1 用常量 X[j]，否则用上一层的 out。"""
        def src(c):
            if li == 0 or j < S - 1:
                return ("const", (li, "x", j, c), X[j][c])
            return ("key", (li - 1, "out", c))
        return src

    def last_src(c):
        if li == 0:
            return ("const", (li, "xlast", c), X[S - 1][c])
        return ("key", (li - 1, "out", c))

    # q / k_j / v_j：每行树形
    mv_rows(units, li, "q", W["Wq"], last_src)
    for j in range(S):
        mv_rows(units, li, "k%d" % j, W["Wk"], mk_src(j))
        mv_rows(units, li, "v%d" % j, W["Wv"], mk_src(j))

    # score：s_j = (q·k_j)/sqrt(d)，点积树形
    def score_terms(j):
        return [("key", (li, "q", i), "key", (li, "k%d" % j, i)) for i in range(d)]
    dot_rows(units, li, "s", S, score_terms,
             post=["op div {key} {key} " + F(sqrt_d)])

    # softmax：mx 组内树形（无跨 unit key）；exp 分元素；sum 树形；p 分组
    u = Unit("softmax_mx")
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
    mx = cur[0]
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

    # a_i = Σ_j p_j v_j_i，树形
    def attn_terms(i):
        return [("key", (li, "p", j), "key", (li, "v%d" % j, i)) for j in range(S)]
    dot_rows(units, li, "a", d, attn_terms)

    # o = Wo a
    mv_rows(units, li, "o", W["Wo"], lambda c: ("key", (li, "a", c)))

    # LN：h_i 分组、mu 树形、c2 分组、var 树形、n_i 分组
    for g0 in range(0, d, GROUP):
        u = Unit("ln_h_%d" % g0)
        for i in range(g0, min(g0 + GROUP, d)):
            if li == 0:
                xi = u.use_const((li, "xlast", i), X[S - 1][i])
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
            "op add {acc} {acc} 0.00001",
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

    # MLP1 relu(W1 n)：树形 + 收尾 relu
    mv_rows(units, li, "u", W["W1"], lambda c: ("key", (li, "n", c)),
            post=["op max {key} {key} 0"])
    # MLP2 y = W2 u
    mv_rows(units, li, "y", W["W2"], lambda c: ("key", (li, "u", c)))
    # out = n + y（分组，避免 d 条串行）
    last_layer = (li == cfg["L"] - 1)
    for g0 in range(0, d, GROUP):
        u = Unit("out_%d" % g0)
        for i in range(g0, min(g0 + GROUP, d)):
            if last_layer:
                u.add("op add out%d %s %s" % (i, u.use((li, "n", i)), u.use((li, "y", i))))
                u.final = i
            else:
                u.add("op add %s %s %s" % (u.produce((li, "out", i)),
                                           u.use((li, "n", i)), u.use((li, "y", i))))
        units.append(u)
    return units


class RigChain(base.Rig):
    def __init__(self, cfg):
        self.cfg = cfg
        self.W, self.g, self.b, self.X = base.gen_weights(cfg)
        self.units = []
        for li in range(cfg["L"]):
            self.units += layer_units_chain(cfg, li, self.W, self.g, self.b, self.X)

# ------------------------------------------------------- 诊断
def chunk_depth(rig):
    """rig.build(cap) 之后的 chunk 依赖 DAG 深度。
    返回 dict(chunks, depth, dep_edges)。depth 单位 = chunk 层数，
    每层在 mlog 里至少要 1 个 tick（块更新顺序好时可级联，实测 3 tick 出值）。"""
    chunks = rig.chunks
    prod_chunk = {}
    for ci, ch in enumerate(chunks):
        for u in ch:
            for k in u.produces:
                prod_chunk.setdefault(k, ci)
    dep = [set() for _ in chunks]
    for ci, ch in enumerate(chunks):
        for u in ch:
            for k in u.uses:
                p = prod_chunk.get(k)
                if p is not None and p != ci:
                    dep[ci].add(p)
    dp = [0] * len(chunks)
    for ci in range(len(chunks)):
        dp[ci] = (1 + max(dp[p] for p in dep[ci])) if dep[ci] else 1
    return dict(chunks=len(chunks), depth=max(dp),
                dep_edges=sum(len(s) for s in dep))
