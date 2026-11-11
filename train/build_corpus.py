# -*- coding: utf-8 -*-
"""build_corpus.py -- 把一堆中文文本变成字符级语料 + 词表。

输入是任意目录（默认仓库根的 corpus/，自己没有就随便指一个），递归收
.md / .txt；清洗成「只留 CJK 汉字 + 中文标点」的字符流，按出现频率切词表，
再按文件划分 train / val。

用法:
    python train/build_corpus.py --texts corpus/ --V 4096
    python train/build_corpus.py --texts corpus/ --V 32 --drop-oov --tag toy

产物（默认落在本目录）:
    corpus.npz     tokens_train / tokens_val (int32) + seg_train / seg_val
    vocab.json     {"id2ch": [...], "ch2id": {...}, "files": [...], "stats": {...}}
  给了 --tag 就是 corpus_<tag>.npz / vocab_<tag>.json。

清洗规则:
  去掉 fenced code block、表格线(| 开头)、YAML front matter、HTML 注释；
  逐字符白名单: CJK 统一表意文字 + 中英文标点白名单；ASCII 字母/数字/符号一律丢弃。
"""
import argparse, json, os, re, sys, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))              # train/
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))          # 仓库根
DEFAULT_TEXTS = os.path.join(ROOT, "corpus")

BT = chr(96)
FENCE = BT * 3
FENCE_RE = re.compile(FENCE + r".*?" + FENCE, re.S)
TILDE_RE = re.compile("~" * 3 + r".*?" + "~" * 3, re.S)

PUNCT = set("。，、；：？！“”‘’（）《》〈〉【】「」『』—…·～")
CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))


def keep(ch):
    o = ord(ch)
    for a, b in CJK_RANGES:
        if a <= o <= b:
            return True
    return ch in PUNCT


def clean_text(raw):
    if raw.startswith("---"):
        m = re.match(r"^---\n.*?\n---\n", raw, re.S)
        if m:
            raw = raw[m.end():]
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = FENCE_RE.sub("\n", raw)
    raw = TILDE_RE.sub("\n", raw)
    out = []
    for line in raw.split("\n"):
        st = line.strip()
        if re.match(r"^[-=*\s|]+$", st) and len(st) > 2:
            continue
        if st.startswith("|"):
            # 表格行：按单元格拆开，保留其中的中文，丢掉 | 与分隔线
            cells = [c.strip() for c in st.strip("|").split("|")]
            line = "".join(c for c in cells if c and not re.match(r"^[-:\s]+$", c))
        buf = []
        for ch in line:
            if keep(ch):
                buf.append(ch)
            elif ch in " \t":
                buf.append(" ")
        seg = "".join(buf).strip()
        seg = re.sub(r" {2,}", " ", seg).strip()
        if seg:
            out.append(seg)
    return "\n".join(out)


def collect_files(root, exts=(".md", ".txt", ".markdown")):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv")]
        for fn in sorted(filenames):
            if fn.lower().endswith(tuple(exts)):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(
        prog="build_corpus.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="从文本目录抽字符级中文语料，切词表，划分 train/val。")
    ap.add_argument("--texts", default=DEFAULT_TEXTS,
                    help="语料目录（递归收 .md/.txt）")
    ap.add_argument("--V", type=int, default=4096, help="词表大小（含 <unk>）")
    ap.add_argument("--out-dir", default=HERE, help="产物目录")
    ap.add_argument("--tag", default="", help="给产物加后缀：corpus_<tag>.npz / vocab_<tag>.json")
    ap.add_argument("--drop-oov", action="store_true",
                    help="词表外的字直接删（并把文本切成若干邻接段），不映射到 <unk>")
    ap.add_argument("--val-mod", type=int, default=10,
                    help="每 val-mod 个文件留一个做 val")
    ap.add_argument("--min-seg", type=int, default=10,
                    help="drop-oov 模式下保留的连续段最短长度（避免窗口跨过被删掉的 OOV）")
    a = ap.parse_args()

    if not os.path.isdir(a.texts):
        raise SystemExit("语料目录不存在: %s\n"
                         "建一个（.md/.txt 随便放），或用 --texts 指到别处。" % a.texts)
    files = collect_files(a.texts)
    if not files:
        raise SystemExit("目录里没找到 .md/.txt: %s" % a.texts)
    docs = []
    for i, p in enumerate(files):
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            t = clean_text(fh.read())
        docs.append((os.path.relpath(p, a.texts).replace("\\", "/"), t,
                     (i % a.val_mod) == (a.val_mod - 1)))

    # 小语料兜底：文件数 >= 2 时至少留一个做 val，否则训练侧没有验证集
    if not any(v for _, _, v in docs):
        if len(docs) >= 2:
            n0, t0, _ = docs[-1]
            docs[-1] = (n0, t0, True)
        else:
            print("build_corpus: 只有 %d 个文件，没有留出验证集（训练时 val 会跳过）"
                  % len(docs), file=sys.stderr)

    freq = {}
    for name, t, is_val in docs:
        if is_val:
            continue
        for ch in t:
            if keep(ch):
                freq[ch] = freq.get(ch, 0) + 1
    order = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    nch = a.V - 1
    id2ch = ["<unk>"] + [c for c, _ in order[:nch]]
    ch2id = {c: i for i, c in enumerate(id2ch) if i > 0}

    # 分段编码：drop_oov 时，被删掉的 OOV 字符会把一段文字切成若干「真邻接」的连续段。
    # 只保留长度 >= min_seg（非 drop 模式下 =1）的段，并记录 seg id；
    # 训练时窗口只在同一段内部取，保证 bigram 邻接是原文真实的邻接。
    def encode(t, sid0, min_len):
        ids, segs = [], []
        dropped = unk = 0
        sid = sid0
        run = []
        for ch in t:
            if not keep(ch):
                continue
            j = ch2id.get(ch)
            if j is None and a.drop_oov:
                dropped += 1
                if min_len < 1:
                    continue          # 不切段：直接丢弃，保持流连续
                if len(run) >= min_len:
                    ids.extend(run)
                    segs.extend([sid] * len(run))
                    sid += 1
                run = []
                continue
            if j is None:
                unk += 1
                run.append(0)
            else:
                run.append(j)
        if len(run) >= min_len:
            ids.extend(run)
            segs.extend([sid] * len(run))
            sid += 1
        return ids, segs, dropped, unk, sid

    tr, va, sg_tr, sg_va = [], [], [], []
    dropped = unk = 0
    sid = 0
    # min_seg=0 表示不按 OOV 切段（整篇文档当一段，只在窗口处不跨文档）
    min_len = a.min_seg if a.drop_oov else 1
    for name, t, is_val in docs:
        ids, segs, d, u, sid = encode(t, sid, min_len)
        dropped += d
        unk += u
        if is_val:
            va.extend(ids)
            sg_va.extend(segs)
        else:
            tr.extend(ids)
            sg_tr.extend(segs)

    import numpy as np
    suffix = ("_" + a.tag) if a.tag else ""
    os.makedirs(a.out_dir, exist_ok=True)
    corpus_path = os.path.join(a.out_dir, "corpus%s.npz" % suffix)
    np.savez(corpus_path,
             tokens_train=np.array(tr, dtype=np.int32),
             tokens_val=np.array(va, dtype=np.int32),
             seg_train=np.array(sg_tr, dtype=np.int32),
             seg_val=np.array(sg_va, dtype=np.int32))

    tot = len(tr) + len(va) + dropped
    vocab = {
        "id2ch": id2ch,
        "ch2id": ch2id,
        "V": a.V,
        "drop_oov": bool(a.drop_oov),
        "files": [{"path": n, "val": v, "chars": len(t)} for n, t, v in docs],
        "stats": {
            "train_tokens": len(tr),
            "val_tokens": len(va),
            "char_types": len(freq),
            "oov_tokens": dropped,
            "oov_rate": round(dropped / max(1, tot), 5),
            "unk_tokens": unk,
            "unk_rate": round(unk / max(1, len(tr) + len(va)), 5),
            "segments": int(sid),
            "top32": "".join(id2ch[1:33]),
            "coverage": round(sum(freq[c] for c in id2ch[1:] if c in freq) / max(1, sum(freq.values())), 5),
        },
    }
    vocab_path = os.path.join(a.out_dir, "vocab%s.json" % suffix)
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"corpus": corpus_path, "vocab": vocab_path, "V": a.V,
                      **vocab["stats"], "files": len(docs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
