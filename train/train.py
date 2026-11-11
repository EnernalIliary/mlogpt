# -*- coding: utf-8 -*-
"""train.py -- 训练 mlogpt-zh-0.72m：numpy 手写反向 + Adam。

"无表嵌入"(table-free embedding)：输入嵌入不是可训练的 E[V][d]，而是固定解析式

    a_i = 0.017 * (1 + i) ** 0.7          # 按维度递增
    b_i = 2.399963229728653 * (i + 1)     # 黄金角乘子
    X[t][i] = sin(a_i * t + b_i)          # t = token id, i = 维下标

（随机傅里叶特征，token 间近似不相关；游戏内逐位可复现，不需要 4096x128 的 world-cell。）

模型（单头注意力 + LayerNorm + ReLU MLP，L 层，只有最后一个位置跨层流动）：

    h = X[S-1]
    for li in range(L):
        ctx = stack([X[j] for j in range(S-1)] + [h])
        K = Wk @ ctx.T ; Vv = Wv @ ctx.T ; q = Wq @ h
        s = (K.T @ q) / sqrt(d) ; p = softmax(s)
        a = p @ Vv.T ; o = Wo @ a
        hh = h + o
        mu = hh.mean() ; var = ((hh-mu)**2).mean()
        n = (hh-mu)/sqrt(var+1e-5)*g + b
        u = relu(W1 @ n) ; y = W2 @ u
        h = n + y
    logits = Wout @ h

可训练参数：Wq/Wk/Wv/Wo (d,d) + W1 (4d,d) + W2 (d,4d) + g/b (d) + Wout (V,d)
= 721,152 个（d=128, V=4096）。

语料：先跑 train/build_corpus.py 把一堆中文文本变成 corpus.npz + vocab.json，
默认都在本目录（--data-dir 可改）。

用法:
    python train/train.py --steps 12000
    python train/train.py --steps 12000 --max-seconds 2700 --resume
"""
import argparse, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))              # train/
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))          # 仓库根
EPS = 1e-5

CFG = dict(d=128, V=4096, L=4, S=16, batch=128, lr=2e-3, val_windows=800)

# 嵌入公式参数（与 mlog 逐位对齐，勿改）
EMB_A_SCALE = 0.017
EMB_A_POW = 0.7
EMB_B_STEP = 2.399963229728653

KEYS = ["Wq", "Wk", "Wv", "Wo", "W1", "W2", "g", "b", "Wout"]


# ---------------------------------------------------------------- 无表嵌入
def emb_coef(d):
    i = np.arange(d, dtype=np.float64)
    a = EMB_A_SCALE * np.power(1.0 + i, EMB_A_POW)
    b = EMB_B_STEP * (i + 1.0)
    return a, b


def emb_table(V, d, dt=np.float64):
    """E0[t, i] = sin(a_i * t + b_i)，float64。运行时按 id 查这一份只读表即可。"""
    a, b = emb_coef(d)
    t = np.arange(V, dtype=np.float64)[:, None]
    return np.sin(t * a[None, :] + b[None, :]).astype(dt)


# ---------------------------------------------------------------- 初始化
def init_params(cfg, seed, E0):
    rng = np.random.default_rng(seed)
    d, V = cfg["d"], cfg["V"]
    sq = 1.0 / math.sqrt(d)
    P = {
        "Wq": rng.normal(0.0, sq, (d, d)),
        "Wk": rng.normal(0.0, sq, (d, d)),
        "Wv": rng.normal(0.0, sq, (d, d)),
        "Wo": rng.normal(0.0, sq, (d, d)),
        "W1": rng.normal(0.0, math.sqrt(2.0 / d), (4 * d, d)),
        "W2": rng.normal(0.0, math.sqrt(2.0 / (4 * d)), (d, 4 * d)),
        "g": np.ones(d),
        "b": np.zeros(d),
        "Wout": rng.normal(0.0, sq, (V, d)),
    }
    return {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in P.items()}


# ---------------------------------------------------------------- 前向 + 反向
def fwd_bwd(P, toks, cfg, E0, need_grad=True):
    """toks: (B, S+1) int。X = 无表嵌入(toks[:, :S])，label = toks[:, S]。"""
    d, S, L, V = cfg["d"], cfg["S"], cfg["L"], cfg["V"]
    B = toks.shape[0]
    BS = B * S
    Bs = math.sqrt(float(d))
    Wq, Wk, Wv, Wo = P["Wq"], P["Wk"], P["Wv"], P["Wo"]
    W1, W2, g, b = P["W1"], P["W2"], P["g"], P["b"]

    idx = toks[:, :S]
    X = E0[idx]                                       # (B,S,d) 固定解析嵌入
    h = X[:, S - 1, :].copy()                         # (B,d)
    ctx = np.concatenate([X[:, :S - 1, :], h[:, None, :]], axis=1)   # (B,S,d)

    caches = []
    for li in range(L):
        ctxf = ctx.reshape(BS, d)
        Kf = ctxf @ Wk.T
        Vf = ctxf @ Wv.T
        K3 = Kf.reshape(B, S, d)
        V3 = Vf.reshape(B, S, d)
        q = h @ Wq.T
        s = (K3 * q[:, None, :]).sum(-1) / Bs
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=1, keepdims=True)
        a = (p[:, :, None] * V3).sum(1)
        o = a @ Wo.T
        hh = h + o
        mu = hh.mean(axis=1, keepdims=True)
        xh = hh - mu
        var = (xh * xh).mean(axis=1, keepdims=True)
        istd = 1.0 / np.sqrt(var + EPS)
        xhat = xh * istd
        n = xhat * g + b
        z = n @ W1.T
        u = np.maximum(z, 0.0)
        y = u @ W2.T
        hn = n + y
        caches.append(dict(ctx=ctx, K3=K3, V3=V3, q=q, p=p, a=a, xh=xh, var=var,
                           istd=istd, xhat=xhat, n=n, z=z, u=u, h_prev=h))
        h = hn
        if li != L - 1:
            ctx = np.concatenate([X[:, :S - 1, :], h[:, None, :]], axis=1)

    logits = h @ P["Wout"].T
    if toks.shape[1] <= S:            # 生成路径：窗口只有 S 个 token，没有 label
        return 0.0, logits, None, h
    le = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(le)
    Z = ex.sum(axis=1, keepdims=True)
    logp = le - np.log(Z)
    tgt = toks[:, S]
    loss = -logp[np.arange(B), tgt].mean()

    if not need_grad:
        return float(loss), logits, None, h

    G = {k: np.zeros_like(P[k]) for k in KEYS}
    dlogits = ex / Z
    dlogits[np.arange(B), tgt] -= 1.0
    dlogits /= B
    G["Wout"] += dlogits.T @ h
    dh = dlogits @ P["Wout"]

    for li in range(L - 1, -1, -1):
        c = caches[li]
        ctx, K3, V3, q, p = c["ctx"], c["K3"], c["V3"], c["q"], c["p"]
        xh, var, istd, xhat = c["xh"], c["var"], c["istd"], c["xhat"]
        n, z, u = c["n"], c["z"], c["u"]
        ctxf = ctx.reshape(BS, d)

        dn = dh.copy()
        G["W2"] += dh.T @ u
        du = dh @ W2
        dz = du * (z > 0.0)
        G["W1"] += dz.T @ n
        dn += dz @ W1

        G["g"] += (dn * xhat).sum(axis=0)
        G["b"] += dn.sum(axis=0)
        dz0 = dn * g
        dhh = istd * (dz0 - dz0.mean(axis=1, keepdims=True)
                      - xhat * (dz0 * xhat).mean(axis=1, keepdims=True))

        dh = dhh.copy()
        G["Wo"] += dhh.T @ c["a"]
        da = dhh @ Wo
        dp = (V3 * da[:, None, :]).sum(axis=2)
        dV3 = p[:, :, None] * da[:, None, :]
        ds = p * (dp - (p * dp).sum(axis=1, keepdims=True)) / Bs
        dK3 = ds[:, :, None] * q[:, None, :]
        dq = (K3 * ds[:, :, None]).sum(1)

        dKf = dK3.reshape(BS, d)
        dVf = dV3.reshape(BS, d)
        G["Wk"] += dKf.T @ ctxf
        G["Wv"] += dVf.T @ ctxf
        dctxf = dKf @ Wk
        dctxf += dVf @ Wv
        G["Wq"] += dq.T @ c["h_prev"]
        dh += dq @ Wq

        # 嵌入固定：dctx 对 X 的梯度没有可训练参数承接，直接丢弃。
        dh += dctxf.reshape(B, S, d)[:, S - 1, :]

    return float(loss), logits, G, h


# ---------------------------------------------------------------- 优化器
class Adam:
    def __init__(self, P, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(P[k]) for k in KEYS}
        self.v = {k: np.zeros_like(P[k]) for k in KEYS}

    def step(self, P, G, lr):
        b1, b2, eps = self.b1, self.b2, self.eps
        for k in KEYS:
            gk = G[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            P[k] -= lr * (self.m[k] / (1 - b1)) / (np.sqrt(self.v[k] / (1 - b2)) + eps)


def clip_grad_(G, maxnorm):
    tot = 0.0
    for k in KEYS:
        tot += float((G[k] * G[k]).sum())
    nrm = math.sqrt(tot)
    if nrm > maxnorm and nrm > 0:
        sc = maxnorm / (nrm + 1e-12)
        for k in KEYS:
            G[k] *= sc
    return nrm


def lr_at(step, target, base, warmup=60):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, target - warmup)
    t = min(1.0, t)
    return base * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * t)))


# ---------------------------------------------------------------- 数据
def load_tokens(path):
    z = np.load(path)
    tr = z["tokens_train"].astype(np.int64)
    va = z["tokens_val"].astype(np.int64)
    st_ = z["seg_train"].astype(np.int64) if "seg_train" in z else np.zeros(len(tr), np.int64)
    sv_ = z["seg_val"].astype(np.int64) if "seg_val" in z else np.zeros(len(va), np.int64)
    return tr, va, st_, sv_


def window_starts(seg, S):
    n = len(seg) - S
    if n <= 0:
        return np.zeros(0, np.int64)
    return np.flatnonzero(seg[:n] == seg[S:S + n])


def make_batch(tok, cfg, rng, starts):
    S = cfg["S"]
    st = starts[rng.integers(0, len(starts), size=cfg["batch"])]
    return np.stack([tok[i:i + S + 1] for i in st])


def eval_loss(P, tok, cfg, nwin, starts, E0, seed=999):
    rng = np.random.default_rng(seed)
    S = cfg["S"]
    nwin = min(nwin, len(starts))
    st = starts[rng.integers(0, len(starts), size=nwin)]
    if nwin == 0:
        return float("nan")           # 没有验证窗口（语料太小 / 没留 val 文件）
    tot, cnt = 0.0, 0
    for i in range(0, nwin, 256):
        b = np.stack([tok[j:j + S + 1] for j in st[i:i + 256]])
        l = fwd_bwd(P, b, cfg, E0, need_grad=False)[0]
        tot += l * len(b)
        cnt += len(b)
    return tot / cnt


# ---------------------------------------------------------------- 主流程
def save_ckpt(path, P, opt, step):
    d = {("p_" + k): P[k] for k in KEYS}
    d.update({("m_" + k): opt.m[k] for k in KEYS})
    d.update({("v_" + k): opt.v[k] for k in KEYS})
    d["step"] = np.array([step], dtype=np.int64)
    np.savez(path, **d)


def load_ckpt(path, P, opt):
    z = np.load(path)
    for k in KEYS:
        P[k] = np.ascontiguousarray(z["p_" + k])
        opt.m[k] = z["m_" + k]
        opt.v[k] = z["v_" + k]
    return int(z["step"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-seconds", type=float, default=2700.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--data-dir", default=HERE,
                    help="语料 / 断点 / 日志目录")
    ap.add_argument("--corpus", default=None, help="corpus.npz（build_corpus.py 的产物）")
    ap.add_argument("--vocab", default=None, help="vocab.json（只记进 meta，训练用不到）")
    ap.add_argument("--out", default=None, help="训练好的权重 npz")
    ap.add_argument("--ckpt", default=None, help="断点 npz（--resume 时读、每 --save-every 写）")
    a = ap.parse_args()
    a.corpus = a.corpus or os.path.join(a.data_dir, "corpus.npz")
    a.vocab = a.vocab or os.path.join(a.data_dir, "vocab.json")
    a.out = a.out or os.path.join(a.data_dir, "weights.npz")
    a.ckpt = a.ckpt or os.path.join(a.data_dir, "ckpt.npz")
    a.loss = os.path.join(a.data_dir, "loss.json")
    a.meta = os.path.join(a.data_dir, "meta.json")

    cfg = dict(CFG)
    V, d, S, L = cfg["V"], cfg["d"], cfg["S"], cfg["L"]
    if a.batch:
        cfg["batch"] = a.batch
    base_lr = a.lr if a.lr else cfg["lr"]

    E0 = emb_table(V, d)
    tr, va, sgt, sgv = load_tokens(a.corpus)
    starts_tr = window_starts(sgt, S)
    starts_va = window_starts(sgv, S)
    if len(starts_tr) == 0:
        starts_tr = np.arange(len(tr) - S - 1, dtype=np.int64)
    if len(starts_va) == 0:
        starts_va = np.arange(len(va) - S - 1, dtype=np.int64)
    has_val = len(va) >= S + 2 and len(starts_va) > 0
    if not a.quiet:
        print("windows: train %d / val %d (seg-masked)" % (len(starts_tr), len(starts_va)), flush=True)
        if not has_val:
            print("  ! 没有可用的验证集（val_tokens=%d）；--val-every 的验证会跳过。"
                  "\n    多放几个文件让 build_corpus.py 能留出一个 val 文件，或调小 --val-mod。"
                  % len(va), file=sys.stderr, flush=True)

    P = init_params(cfg, a.seed, E0)
    opt = Adam(P, lr=base_lr)
    step = 0
    ck = a.ckpt
    if a.resume and os.path.exists(ck):
        step = load_ckpt(ck, P, opt)
        if not a.quiet:
            print("resumed from %s at step %d" % (ck, step), flush=True)

    hist = {"step": [], "train": [], "val": []}
    lp = a.loss
    if a.resume and os.path.exists(lp):
        old = json.load(open(lp, "r", encoding="utf-8"))
        for k in hist:
            hist[k] = old.get(k, [])[:]

    rng = np.random.default_rng(a.seed + 7919 + step)
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    tgt = a.steps
    stopped = "steps"
    while step < tgt:
        if a.max_seconds and (time.time() - t0) > a.max_seconds:
            stopped = "time"
            print("time budget reached at step %d" % step, flush=True)
            break
        b = make_batch(tr, cfg, rng, starts_tr)
        loss, _, G, _ = fwd_bwd(P, b, cfg, E0, need_grad=True)
        gn = clip_grad_(G, 1.0)
        lr = lr_at(step, tgt, base_lr)
        opt.step(P, G, lr)
        run_loss += loss
        run_n += 1
        step += 1
        if step % 100 == 0 and not a.quiet:
            print("step %5d  train %.4f  lr %.2e  gnorm %.2f  %.1fs" % (
                step, run_loss / run_n, lr, gn, time.time() - t0), flush=True)
        if step % a.val_every == 0 or step == tgt:
            vl = eval_loss(P, va, cfg, cfg["val_windows"], starts_va, E0) if has_val else float("nan")
            hist["step"].append(step)
            hist["train"].append(round(run_loss / max(1, run_n), 6))
            hist["val"].append(round(vl, 6) if has_val else None)
            with open(lp, "w", encoding="utf-8") as fh:
                json.dump(hist, fh, ensure_ascii=False)
            run_loss, run_n = 0.0, 0
            if not a.quiet:
                if has_val:
                    print("  val %.4f  ppl %.2f  (%.1fs)" % (vl, math.exp(min(vl, 20)), time.time() - t0), flush=True)
                else:
                    print("  val 跳过（没有验证集）  (%.1fs)" % (time.time() - t0), flush=True)
        if step % a.save_every == 0:
            save_ckpt(ck, P, opt, step)

    vl = eval_loss(P, va, cfg, cfg["val_windows"], starts_va, E0)
    if not hist["step"] or hist["step"][-1] != step:
        hist["step"].append(step)
        hist["train"].append(round(run_loss / max(1, run_n), 6))
        hist["val"].append(round(vl, 6))
    with open(lp, "w", encoding="utf-8") as fh:
        json.dump(hist, fh, ensure_ascii=False)

    save_ckpt(ck, P, opt, step)
    wp = a.out
    os.makedirs(os.path.dirname(os.path.abspath(wp)) or ".", exist_ok=True)
    np.savez(wp, **{k: P[k] for k in KEYS})
    meta = dict(tag="mlogpt", table_free=True, d=d, V=V, L=L, S=S, steps=step,
                stopped=stopped,
                vocab=os.path.basename(a.vocab), corpus=os.path.basename(a.corpus),
                seed=a.seed, batch=cfg["batch"], lr=base_lr,
                val_loss=vl, val_ppl=math.exp(vl), uniform_ppl=float(V),
                train_tokens=int(len(tr)), val_tokens=int(len(va)),
                seconds=round(time.time() - t0, 1),
                params=int(sum(P[k].size for k in KEYS)),
                dtype="float64",
                embed=dict(form="sin(a_i*t + b_i)",
                           a="0.017*(1+i)**0.7", b="2.399963229728653*(i+1)",
                           a_scale=EMB_A_SCALE, a_pow=EMB_A_POW, b_step=EMB_B_STEP,
                           note="t=token id, i=dim index 0..d-1; float64 sin; no trainable E"))
    with open(a.meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
