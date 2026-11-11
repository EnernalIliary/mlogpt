# 复现

本页记录「从零跑出同一个蓝图」的每一步，命令与数字都是在洁净室
（仓库整份复制到另一个目录，只带 Python + numpy）里真跑出来的，不是估算。

## 环境

| 项 | 实测值 |
| --- | --- |
| Python | 3.11.4 (Windows, 64-bit) |
| numpy | 2.4.6 |
| 其它依赖 | 无 —— 只用标准库 + numpy |
| 磁盘 | 全尺寸产物 71 MB，另加同目录的 stats json 约 0.4 MB |

```bash
python -m pip install -r requirements.txt   # 就一行：numpy>=1.24
```

想完全按本文复现，建议每条命令都加 `-X utf8`（Windows 控制台默认不是 UTF-8，
中文 token 解码会花屏）。

## 0. 自检思维

三条命令按「小 → 中 → 大」递进：

1. **玩具构建**（d=8/V=32，约 0.1 s）：先把「权重 → 布线 → 打包 .msch」整条路走通，
   不用等一分钟就知道代码是不是好的。
2. **参考前向**（`mlogpt/reference.py`）：纯 numpy 的同一套前向 + 采样，
   和构建时写进 stats 的 `ref_tokens` 必须逐 token 相同。
3. **全尺寸构建**（d=128/V=4096，约 50–60 s，71 MB）：真模型铺成 10,786 台处理器。

## 1. 玩具构建（约 0.1 s）

```bash
python -X utf8 mlogpt/gen.py --build --d 8 --S 8 --L 2 --V 32 --N 4 --cap 600 \
  --seed-tok 0 \
  --weights weights/toy-d8s8l2v32.npz \
  --vocab weights/toy-vocab-32.json \
  --out toy.msch
```

实测输出（原样）：

```text
W	units 854	0.0s
W	packed 16 chunks	0.0s
W	procs 49 cells 16	0.0s
W	built toy.msch  0.08 MB  49 procs  16 cells  max 579 lines  0.0s
W	ref tokens [5, 12, 12, 3]
```

要点：

- **49 台处理器、16 个 world-cell、单台最多 579 行、0.08 MB**；
- `max 579 lines` 是最终单台最大语句数，离硬上限 1000 还有余量；真撞上硬上限才会
  打印 `!! OVER LIMIT: [...]`；
- 权重只有 12 KB（`weights/toy-d8s8l2v32.npz`），是给这条链路做冒烟用的。

**49 还是 57？** 上面这条命令**不带** `--input`，是 49 台（其中 1 台是负责说明 / 状态面板的
`reporter`）。加上 `--input --input-text 数条` 会多铺 8 台分词器（57 台，仍是 16 个 world-cell、
579 行、0.09 MB），`ref_tokens` 也从「只按 `--seed-tok` 无条件采样」变成
「按 `数条` 的 prompt 采样」，得到 `[2, 12, 9, 3]` —— 正好等于下面第 2 步
`reference.py --text 数条` 的输出。两个数字都对，取决于加不加输入模式：

```bash
python -X utf8 mlogpt/gen.py --build --d 8 --S 8 --L 2 --V 32 --N 4 --cap 600 \
  --seed-tok 0 --input --input-text 数条 \
  --weights weights/toy-d8s8l2v32.npz --vocab weights/toy-vocab-32.json \
  --out toy-in.msch
# W	procs 57 cells 16
# W	built toy-in.msch  0.09 MB  57 procs  16 cells  max 579 lines  0.1s
# W	ref tokens [2, 12, 9, 3]
```

## 2. 参考前向（约 1 s，含解释器启动）

不碰蓝图，直接在 numpy 里跑同一个模型和同一套采样规则：

```bash
python -X utf8 mlogpt/reference.py \
  --weights weights/toy-d8s8l2v32.npz \
  --vocab weights/toy-vocab-32.json \
  --text 数条 --N 4
```

实测输出：

```text
text    : 数条
prompt  : [15, 21, 0, 0, 0, 0, 0, 0]
tokens  : [2, 12, 9, 3]
decoded : （—不）
```

后半段的 `0` 是补位（窗口 S=8，prompt 只占前两位），采样直接吃这 8 个位置。

## 3. 全尺寸构建（约 50–60 s，71 MB）

契约命令（`_FACTS.md` 里那条）：

```bash
python -X utf8 mlogpt/gen.py --build --d 128 --S 16 --L 4 --V 4096 --N 8 --cap 900 \
  --seed-tok 82 --table-free --input --input-text 量持法 \
  --weights weights/mlogpt-zh-0.72m-fp32.npz \
  --vocab weights/vocab-zh-4096.json \
  --out mlogpt-zh-0.72m.msch
```

实测输出（原样，阶段耗时）：

```text
W	units 216124	10.8s
W	packed 8098 chunks	11.8s
W	split 1010 chunks (round 0) -> 9108	12.8s
W	procs 10786 cells 170	21.3s
W	built mlogpt-zh-0.72m.msch  71.32 MB  10786 procs  170 cells  max 983 lines  48.4s
W	ref tokens [45, 390, 400, 18, 196, 45, 2, 127]
```

同目录会写一份 `mlogpt-zh-0.72m.json`（stats，约 0.4 MB）。实测字段：

| 字段 | 值 |
| --- | --- |
| `nprocs` / `ncells` | 10786 / 170 |
| `nworldcell` | 169 |
| `msg_instr` / `msg_stat` / `msg_in` | 9 / 10 / 11（说明 / 状态 / 输入方块，1 基；null 表示该模式没有） |
| `tiles`（.msch 里的方块总数） | 10966 = 10786 处理器 + 169 world-cell + 11 message |
| `units` | 216124 |
| `nfwd` / `nhead` / `nsplit` | 9108 / 1366 / 1010 |
| `nval` / `nvalcell` / `nacell` | 64044 / 126 / 4 |
| `total_stmts` / `fwd_stmts` | 8756803 / 7362930 |
| `maxlines` | 983（单台最大语句数，低于硬上限 1000；`over` 为空数组） |
| `bytes` | 71320323 |
| `build_s` | 48.4（同一台机器首跑 56.6，受缓存/负载影响） |
| `ref_tokens` | [45, 390, 400, 18, 196, 45, 2, 127] |
| `input` / `input_idx` | true / 10（`msgpos` 的下标，指向最右那块输入方块） |
| `prompt` | [30, 527, 101, 0, 0, …] |
| `layout` | `{"cc": 13, "ncellrows": 13, "y0": 15, "pw": 104}` |

产物大小 `71320323` 字节 = 71.32 MiB，和 `ls` / `stat` 对得上；两次独立构建的
`bytes` / `maxlines` / `ref_tokens` 完全一致（只有 `build_s` 在 48–57 s 之间浮动）。

同一条模型，numpy 端的解码结果：

```bash
python -X utf8 mlogpt/reference.py \
  --weights weights/mlogpt-zh-0.72m-fp32.npz \
  --vocab weights/vocab-zh-4096.json \
  --text 量持法 --N 8
```

```text
text    : 量持法
prompt  : [30, 527, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tokens  : [45, 390, 400, 18, 196, 45, 2, 127]
decoded : 同任意只相同（所
```

`tokens` 和上面构建日志里的 `ref_tokens` 完全一致 —— 这就是「蓝图里的模型」
和「numpy 里的模型」是同一个东西的证据。注意这只是 0.72 M 参数、24.6 万字语料
训出来的玩具规模模型，输出不成句是正常的（见 `docs/05-limits.md`）。

## 4. 网页控制台（可选）

```bash
python -X utf8 web/ui.py                 # http://127.0.0.1:8771/
python -X utf8 web/ui.py --work out/ --port 9000 --no-open
```

页面上「构建」按钮就是第 1/3 步那条命令（参数可改），产物默认写到 `--work`
目录（默认 `out/`）。「运行」按钮要走下面第 5 步的外部模拟器。

## 5. 在游戏里真跑（可选，需要仓库外的东西）

```bash
python -X utf8 mlogpt/gen.py --run mlogpt-zh-0.72m.msch --ticks 120 --input-text 量持法
```

这条路**不是**纯 Python 能跑完的，需要：

1. **Mindustry v160.7** 的 server jar（`desktop-160.7-release.jar` 之类）与 **JRE 21**；
2. 姊妹仓 [mlog-agentskill](https://github.com/EnernalIliary/mlog-agentskill) 里的
   `tools/mlogsim.py`（把 jar 里的 mlog 解释器拉起来跑 tick 的封装）；
3. `mlogsim` 能被 `import`（把它所在目录加进 `PYTHONPATH`，或直接 `pip install -e` 那个仓）。

缺这些时 `--run` 会在启动阶段直接退出并打印上面这段指引（不是崩溃堆栈）：

```text
运行侧需要 Mindustry 模拟器：
  1) 装 Mindustry v160.7 server jar + JRE21
  2) 从 https://github.com/EnernalIliary/mlog-agentskill 取 tools/mlogsim.py
  3) 让它能在 PYTHONPATH 里被 import（或把那份 tools/ 加进 PYTHONPATH）
只想构建蓝图的话不需要这些，去掉 --run 即可。
```

跑通时，游戏内 10,786 台处理器装载约 45 s，`量持法` 出 8 个 token 约 20 tick；
逐 token 与 `reference.py` 一致（不一致就是布线/编码 bug，不是模型问题）。

## 6. 训练（可选）

训练不是复现蓝图的必要条件（权重已在 `weights/` 里），但想重训：

```bash
# 1) 自己准备一个语料目录（.md / .txt），例如 ./corpus/
python -X utf8 train/build_corpus.py --texts corpus/ --V 4096
#   -> train/corpus.npz, train/vocab.json

# 2) 训练（batch 128、lr 2e-3；步数自己定）
python -X utf8 train/train.py --steps 12000 --data-dir train/
#   -> train/weights.npz, train/loss.json, train/meta.json（每 --save-every 写 train/ckpt.npz）
```

冒烟过的最小闭环（本文真跑过）：两段中文、`--V 256` 建语料 → `--steps 20`
训练，3.1 s 跑完，写出 `weights.npz` / `loss.json` / `meta.json` / `ckpt.npz`；
val 从 6.36 降到 1.93。12000 步的时长取决于机器，本页不编造。

本仓权重的来历：12000 步后 val_ppl 94.8、top-1 18.5%；语料 24.6 万字，
train 224,714 / val 21,646 token。语料本身没有随仓库分发（版权原因），
所以重训只能得到「同规模、不同语料」的权重，token 也不会和上面的对拍数字一样。

## 一个刻意的遗留命名

`.msch` 里的蓝图标题仍然叫 `closure_wpl_d128s16l4v4096`（`mlogpt/gen.py:959`）。
这是研究阶段的名字，改掉它会让产物的字节数与本文档记录的 71,320,323 对不上
（蓝图名进的是压缩块），所以保留原名。它只是蓝图标题，不影响生成逻辑。
在 Mindustry 里导入后看到的蓝图名就是它。

## 常见坑

- **71 MB 的 `.msch` 不进 git**（`.gitignore` 已忽略 `*.msch`）。构建产物只留在本地；
  仓库里带的只有 5.5 MB 权重和两个词表。
- **`--cap` 与 `maxlines` 不是一回事**：`--cap` 是「把单元打包成处理器」时每包的
  语句预算（默认 900），之后还有一个硬上限 `LIMIT = 1000` 兜底；`maxlines` 是最终
  单台最大语句数（这里是 983，卡在两者之间）。`stats.over` 列出超过硬上限的处理器，
  空数组就代表全都在 1000 行以内。把 `--cap` 调小会拆出更多处理器（产物更大），
  调大则更容易撞上硬上限拆分。
- **`--ticks` 只在 `--run` 下有意义**，它是「总共跑多少 tick」（覆盖 `--max-ticks`）；
  构建阶段传了也会被忽略，别把它当成「生成几个 token」（那是 `--N`）。
- **中文参数**：Windows 控制台用 GBK，命令行直接传中文可能乱码，务必
  `python -X utf8`；从 PowerShell 调时用 `& "C:\path\to\python.exe"` 这种调用符。
- **`--table-free`** 决定游戏内是否用 sin 现算嵌入（省 1024 台嵌入服务器）；
  真权重 `weights/mlogpt-zh-0.72m-fp32.npz` 里没有 `E` 键，所以必须开；
  玩具权重 `weights/toy-d8s8l2v32.npz` 里有 `E`（32×8），**别**对玩具开 `--table-free`。
  `reference.py` 会按 npz 里有没有 `E` 自动判断，不用手动传。
- **`--out` 默认**写当前目录的 `mlogpt_d<d>s<S>l<L>v<V>.msch`；想要干净目录就显式传
  `--out out/xxx.msch`（`out/` 不存在时会自动建）。
- **stats 文件名**跟 `.msch` 走：`mlogpt-zh-0.72m.msch` 的 stats 是
  `mlogpt-zh-0.72m.json`（去掉 `.msch` 后缀再加 `.json`），不是 `.msch.json`。
