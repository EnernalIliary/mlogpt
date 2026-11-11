# -*- coding: utf-8 -*-
"""reference.py -- numpy 参考实现：前向、采样、权重加载，外加一个对拍用的 CLI。

gen.py 生成的 mlog 程序是照这个实现逐条展开的（同样的权重、同样的运算顺序），
所以构建产物 stats.json 里的 ref_tokens 就是这里算出来的。游戏里跑出别的序列，
就说明 mlog 那边有 bug —— 这就是整套东西的判据。

采样口径（和游戏内完全一致，改之前先想清楚）：
  * 单头注意力 + pre-LN + 残差，无嵌入表时 X[t][i] = sin(a_i*t + b_i)
  * logits = Wout @ h，温度 1.0
  * w = exp(z)，**不减 max**（游戏里逐条累加，减 max 就会不同序）
  * thr = r[t] * sum(w)，从 j=0 往后串行累加，第一个 cum >= thr 的 j 胜出

用法：
  # 直接给权重和词表（d/V 从权重形状读，S/L 从文件名里的 d<d>s<S>l<L>v<V> 读）
  python mlogpt/reference.py --weights weights/mlogpt-zh-0.72m-fp32.npz \\
        --vocab weights/vocab-zh-4096.json --text 量持法 --N 8

  # 更稳的方式：直接读构建产物旁边的 stats.json，d/S/L/V/prompt 全部照抄
  python mlogpt/reference.py --stats mlogpt-zh-0.72m.json

  # 玩具模型（文件名 d8s8l2v32 自带配置）
  python mlogpt/reference.py --weights weights/toy-d8s8l2v32.npz \\
        --vocab weights/toy-vocab-32.json --text 数条 --N 4
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "mlogpt-zh-0.72m-fp32.npz")
DEFAULT_VOCAB = os.path.join(ROOT, "weights", "vocab-zh-4096.json")

LN_EPS = 0.00001


# ---------------------------------------------------------------- 数值核心
def emb_params(d):
    """无表嵌入（B 路线）：X[t][i] = sin(a_i*t + b_i)"""
    a = [0.017 * (1 + i) ** 0.7 for i in range(d)]
    b = [2.399963229728653 * (i + 1) for i in range(d)]
    return a, b


def emb_row(tok, d):
    a, b = emb_params(d)
    return np.sin(np.array(a) * float(tok) + np.array(b))


def ref_forward(Wt, X, cfg):
    d, S, L = cfg["d"], cfg["S"], cfg["L"]
    h = X[S - 1].copy()
    for li in range(L):
        ctx = np.stack([X[j] for j in range(S - 1)] + [h])
        K = Wt["Wk"] @ ctx.T
        Vv = Wt["Wv"] @ ctx.T
        q = Wt["Wq"] @ h
        s = (K.T @ q) / math.sqrt(d)
        p = np.exp(s - s.max())
        p = p / p.sum()
        a = p @ Vv.T
        o = Wt["Wo"] @ a
        hh = h + o
        mu = hh.mean()
        var = ((hh - mu) ** 2).mean()
        n = (hh - mu) / math.sqrt(var + LN_EPS) * Wt["g"] + Wt["b"]
        u = np.maximum(Wt["W1"] @ n, 0.0)
        y = Wt["W2"] @ u
        h = n + y
    return h


def sample_ref(Wt, cfg):
    """与 mlog 同序：exp 不减 max、串行累加、第一个 cum>=thr。"""
    d, S, V, N, TEMP = cfg["d"], cfg["S"], cfg["V"], cfg["N"], cfg["TEMP"]
    tf = bool(cfg.get("table_free"))
    E = None if tf else Wt["E"]

    def row(tok):
        return emb_row(tok, d) if tf else E[tok]

    pr = [int(x) for x in (cfg.get("prompt") or [])][:S]
    if pr or cfg.get("input"):
        pr = pr + [0] * (S - len(pr))
        win = pr[:S]
    else:
        win = [cfg["seed_tok"]] * S
    out, allz = [], []
    for t in range(N):
        X = np.stack([row(tok) for tok in win])
        h = ref_forward(Wt, X, cfg)
        z = (Wt["Wout"] @ h) / TEMP
        w = np.exp(z)
        tot = 0.0
        for x in w:
            tot += float(x)
        thr = float(Wt["r"][t]) * tot
        cum = 0.0
        idx = V - 1
        for j in range(V):
            cum += float(w[j])
            if cum >= thr:
                idx = j
                break
        out.append(int(idx))
        allz.append([float(z[idx]), float(w[idx]), float(tot), float(thr)])
        win = win[1:] + [idx]
    return out, allz


# ---------------------------------------------------------------- 权重 / 词表
def peek_shapes(path):
    """只读 npz 的 d / V / 有没有 E，用来在建蓝图前给出人话报错。"""
    with np.load(path) as z:
        return dict(d=int(z["Wq"].shape[0]), V=int(z["Wout"].shape[0]),
                    has_E=("E" in z.files))


def load_weights(cfg, path):
    """读训练好的权重。table_free=True 时不要求 E（嵌入在游戏内用 sin 现算）。"""
    z = np.load(path)
    tf = bool(cfg.get("table_free"))
    need = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "g", "b", "Wout"]
    if not tf:
        need = ["E"] + need
    miss = [k for k in need if k not in z]
    if miss:
        hint = ""
        if miss == ["E"]:
            hint = ("\n  这个 npz 是无表嵌入(no-E)权重，加 --table-free 再跑；"
                    "玩具权重 weights/toy-d8s8l2v32.npz 里有 E，不要加 --table-free。")
        raise SystemExit("weights missing keys: %s%s" % (miss, hint))
    W = {k: np.asarray(z[k], dtype=float) for k in need}
    if not tf:
        assert W["E"].shape == (cfg["V"], cfg["d"]), W["E"].shape
    assert W["Wout"].shape == (cfg["V"], cfg["d"]), W["Wout"].shape
    if tf:
        # 与训练侧对拍口径一致：r 表由下标直接算出，不存盘
        W["r"] = np.array([(k * 0.6180339887 + 0.31) % 1.0 for k in range(cfg["N"])])
        W["X"] = None
    else:
        rng = np.random.default_rng(cfg["seed"] + 977)
        W["r"] = rng.random(cfg["N"])
        W["X"] = rng.normal(0, 1, (cfg["S"], cfg["d"]))
    return W


def text_to_ids(text, voc):
    """文本 -> token id（与 mlog 分词器同一规则：同码取首个 id，未命中 0）。"""
    code2id = {}
    for k, ch in enumerate(voc):
        if isinstance(ch, str) and len(ch) == 1:
            code2id.setdefault(ord(ch), k)
    return [code2id.get(ord(c), 0) for c in text]


def decode(ids, voc):
    """token id -> 文本；空槽 / 多字符 / 控制符一律显示成 ?。"""
    out = []
    for i in ids:
        ch = voc[i] if 0 <= i < len(voc) else ""
        if not ch or len(ch) != 1 or ch in ('"', "\\", "\n", "\r", "\t"):
            ch = "?"
        out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- 配置
_NAME_RE = re.compile(r"d(\d+)s(\d+)l(\d+)v(\d+)")


def infer_cfg(weights, vocab, d=None, S=None, L=None, V=None, table_free=None):
    """从权重形状 + 文件名推 d/S/L/V；显式参数优先。"""
    z = np.load(weights)
    cfg = dict(d=d or int(z["Wq"].shape[0]),
               V=V or int(z["Wout"].shape[0]),
               S=S or 16, L=L or 4,
               N=4, seed=7, seed_tok=0, TEMP=1.0)
    m = _NAME_RE.search(os.path.basename(weights))
    if m:
        nd, nS, nL, nV = (int(x) for x in m.groups())
        if d is None:
            cfg["d"] = nd
        if S is None:
            cfg["S"] = nS
        if L is None:
            cfg["L"] = nL
        if V is None:
            cfg["V"] = nV
    if table_free is None:
        table_free = "E" not in z.files
    cfg["table_free"] = bool(table_free)
    if vocab and os.path.exists(vocab):
        with open(vocab, "r", encoding="utf-8") as f:
            vo = json.load(f)
        cfg["vocab"] = vo["id2ch"] if isinstance(vo, dict) else vo
    return cfg


def cfg_from_stats(path, vocab=None):
    with open(path, "r", encoding="utf-8") as f:
        st = json.load(f)
    cfg = dict(d=st["d"], S=st["S"], L=st["L"], V=st["V"], N=st["N"],
               seed=st.get("seed", 7), seed_tok=st.get("seed_tok", 0),
               TEMP=st.get("TEMP", 1.0), table_free=bool(st.get("table_free")),
               input=bool(st.get("input")))
    if st.get("prompt"):
        cfg["prompt"] = [int(x) for x in st["prompt"]]
    if vocab and os.path.exists(vocab):
        with open(vocab, "r", encoding="utf-8") as f:
            vo = json.load(f)
        cfg["vocab"] = vo["id2ch"] if isinstance(vo, dict) else vo
    return cfg


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="reference.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="numpy 参考实现：打印采样出的 token 与解码文本。")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--text", action="append", default=[],
                    help="输入文本，可重复给多段")
    ap.add_argument("--input-text", default=None, help="同 --text（单段写法）")
    ap.add_argument("--N", type=int, default=4, help="采多少个 token")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--S", type=int, default=None, help="窗口长度")
    ap.add_argument("--L", type=int, default=None, help="层数")
    ap.add_argument("--V", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seed-tok", type=int, default=0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--table-free", dest="table_free", action="store_true", default=None,
                    help="强制无表嵌入（默认按权重里有没有 E 自动判断）")
    ap.add_argument("--with-emb", dest="table_free", action="store_false",
                    help="强制用权重里的嵌入表 E")
    ap.add_argument("--stats", default=None,
                    help="构建产物旁边的 stats.json；给了就照抄 d/S/L/V/prompt")
    ap.add_argument("--verbose", action="store_true", help="每个 token 打印 z/w/tot/thr")
    ap.add_argument("--json", default=None, help="把结果写成 json")
    a = ap.parse_args(argv)

    if a.stats:
        cfg = cfg_from_stats(a.stats, a.vocab)
        if a.N:
            cfg["N"] = a.N
    else:
        cfg = infer_cfg(a.weights, a.vocab, a.d, a.S, a.L, a.V, a.table_free)
        cfg.update(N=a.N, seed=a.seed, seed_tok=a.seed_tok, TEMP=a.temp)
    W = load_weights(cfg, a.weights)

    texts = list(a.text or [])
    if a.input_text:
        texts.append(a.input_text)
    if not texts and not cfg.get("prompt"):
        texts = [""]
    if cfg.get("prompt") and not texts:
        texts = [None]          # 用 stats 里记的 prompt

    res = []
    for txt in texts:
        c = dict(cfg)
        if txt is not None:
            ids = text_to_ids(txt, cfg.get("vocab") or [])
            c["prompt"] = (ids + [0] * c["S"])[:c["S"]]
            c["input"] = True
        tokens, zinfo = sample_ref(W, c)
        decoded = decode(tokens, cfg.get("vocab") or [])
        print("text    : %s" % ("<stats prompt>" if txt is None else txt))
        print("prompt  : %s" % c.get("prompt"))
        print("tokens  : %s" % tokens)
        print("decoded : %s" % decoded)
        if a.verbose:
            for i, (z, w, tot, thr) in enumerate(zinfo):
                print("  t%-2d z=%.6g w=%.6g tot=%.6g thr=%.6g" % (i, z, w, tot, thr))
        res.append(dict(text=txt, prompt=c.get("prompt"), tokens=tokens,
                        decoded=decoded, z=zinfo))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
