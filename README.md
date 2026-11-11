# mlogpt

**在 Mindustry 的 mlog 逻辑处理器里跑一个 0.72M 参数的中文 Transformer。**

玩家在 `message` 方块里输入中文，10,786 台世界处理器（world processor）组成的阵列逐字生成中文，把结果写回 8 个 `message` 方块。
它是一个**机制演示**：证明一个 Transformer 的前向传播可以被编译成 8,767,542 条 mlog 语句、切成 10,786 块、在游戏里真的跑出和 numpy 逐位一致的 token。
它**不是**一个能聊天的语言模型——原因写在「诚实的局限」里，请先看那一节再决定要不要 star。

> **English abstract.** mlogpt-zh-0.72m is a 721,152-parameter, decoder-only Chinese Transformer whose entire forward pass
> (analytic embedding, single-head attention, FFN, 4096-way output head) is compiled into 8,767,542 mlog statements spread
> over 10,786 Mindustry world processors — a 71.41 MB blueprint that takes about 52 s to build. Fed a short Chinese prompt
> typed into a message block, the array emits characters one at a time (~5 s per character; 8 characters in 60 ticks / 42.7 s
> on a real headless executor), token-exact against the numpy reference. This is a mechanism demo, not a usable language model:
> val perplexity is 94.8 across a 4096-character vocabulary and top-1 accuracy is 18.5%, so output reads like term fragments.
> MIT licensed. Repository: https://github.com/EnernalIliary/mlogpt

---

## 下载（不构建）

不想跑构建的，直接拿成品：**[Release v0.1.1](https://github.com/EnernalIliary/mlogpt/releases/tag/v0.1.1)**（v0.1.1 修掉了轮询自旋，空转快 6.1×，见 [04 §3](docs/04-performance.md)）

| 资产 | 用途 |
| --- | --- |
| `mlogpt-zh-0.72m.msch` | 71,414,679 B。全尺寸蓝图 / 10,966 方块。留出约 104×104 空地，贴进去、按播放，右键**最右边**的 message 方块打中文 |
| `mlogpt-toy-d8s8l2v32.msch` | 87,084 B，玩具版 57 台，10 秒验证「能贴、能跑、能出字」，不需要大空间 |
| `mlogpt-zh-0.72m-fp32.npz` + `vocab-zh-4096.json` | 5,771,362 B + 58,207 B，权重与词表（仓库里也有） |

world processor 只在编辑器 / 沙盒 / `rules.allowEditWorldProcessors` 下运行，战役里贴了不会动。

---

## 一句话定位

把一个小 Transformer 的**全部推理**（嵌入、注意力、FFN、输出层、采样）编译成 mlog，铺进 Mindustry 的处理器阵列，让游戏里的方块自己算出下一个字。

| 是 | 不是 |
| --- | --- |
| 一个可复现的「模型 → 指令流 → 游戏内执行」编译器演示 | 一个能对话、能写作、能干活的模型 |
| 一份 8,767,542 条语句、10,786 台处理器的真实产物 | 一个跑在 GPU 上的玩具 |
| 与 numpy 参考实现逐位对拍的推理结果 | 一个「看着像中文」的随机字符发生器 |

模型叫 `mlogpt-zh-0.72m`：中文、字符级、0.72M 参数、4 层、d=128、窗口 16 个 token。

---

## 它长什么样

游戏里你看到的是 10,786 台 world processor 铺成的大方阵（约 104 × 104 格）。每台处理器里跑着的是一段 mlog 程序，块与块之间靠 memory cell 传数、靠 barrier 对齐。

```
  message11（你打的中文）
        │
        ▼
  144 台分词器          16 个位置 × 9 个 id 段，每段至多 456 个 id
  逐字符比较 → token id
        │
        ▼  token id 落进 memory cell
  前向阵列              9,108 台，7,362,930 条语句
  嵌入 → 4 × 共享 block → 输出层 → 采样
        │              barrier 同步，按 ≤999 条切块，权重编进语句常量
        ▼  4096 维 logits → 1 次比较采样
  message1..8           8 个字，每 4 个换行 → 2 行 × 4 字
  message9 / 10         说明面板（用法）/ 状态面板（启动进度、生成字数、完成提示）
```

构建产物自带 11 个 message 方块：输出 1~8 ｜ 说明 ｜ 状态 ｜ 输入（最右），供游戏内看进度与操作提示。
10,786 台 = 前向 9,108 + 输出头 1,366 + 分词器 144 + 采样/预载/驱动等 168。

整条链上一个 token 要走 7~8 个同步点。这是它慢的真正原因，见 [docs/04-performance.md](docs/04-performance.md)。

---

## 快速开始

### 0. 环境

```bash
git clone https://github.com/EnernalIliary/mlogpt
cd mlogpt
python -m pip install -r requirements.txt   # 只有 numpy>=1.24
```

Python 3，除 numpy 外不需要任何第三方包。训练、构建、参考前向、网页外壳都只用 stdlib + numpy。

### 1. 玩具自检（不需要游戏，几秒钟）

玩具配置 d=8 / S=8 / L=2 / V=32 / N=4，只用来验证「numpy 参考前向」和「mlog 蓝图」两条路算出来的 token 一样。它的输出没有意义。

```bash
python mlogpt/gen.py --build --d 8 --S 8 --L 2 --V 32 --N 4 --cap 600 --seed-tok 0 \
  --weights weights/toy-d8s8l2v32.npz --vocab weights/toy-vocab-32.json --out toy.msch

python mlogpt/reference.py --weights weights/toy-d8s8l2v32.npz \
  --vocab weights/toy-vocab-32.json --text 数条 --N 4
```

第二条命令直接打印 numpy 参考路径采样出的 token id。构建产物里内嵌了一份 `ref`，两份对得上，说明切块、barrier、跨块依赖都没有把数值算错。

玩具权重里带一张 32×8 的嵌入表 `E`（所以玩具构建不传 `--table-free`）；全尺寸构建走解析式嵌入，没有表可查。

### 2. 全尺寸构建（约 1 分钟，71.41 MB）

```bash
python mlogpt/gen.py --build \
  --d 128 --S 16 --L 4 --V 4096 --N 8 --cap 900 --seed-tok 82 \
  --table-free --input --input-text 量持法 \
  --weights weights/mlogpt-zh-0.72m-fp32.npz \
  --vocab  weights/vocab-zh-4096.json --out mlogpt-zh-0.72m.msch
```

约 52 s 后得到 `mlogpt-zh-0.72m.msch`（71.41 MB）和同名 `.json` 统计文件；`.json` 会被 `--run` 和 `web/ui.py` 读走。

71 MB 的蓝图**不在 git 里**（`.gitignore` 里排除 `*.msch`），请自己构建。

### 3. 游戏内导入

1. 把 `mlogpt-zh-0.72m.msch` 放进 Mindustry 的蓝图目录（或用游戏内的「导入蓝图」按钮选择该文件）；
2. 粘贴蓝图，等阵列放完（10,786 台处理器 + 170 个 memory cell）；
3. 在 `message11` 里打字，等 `message1..8` 出字（8 个字约 42.7 s，见下）；
4. 出完一轮后**把 `message11` 清空**，它才会接下一轮。

只在 Mindustry v160.7 上验过：world processor 与 `setrate` 的语义是按这个版本写的。

### 4. headless 跑（需要 jar，最可信的一次验证）

headless 指的是：把蓝图里的 mlog 真丢进 Mindustry 的 v160.7 执行器里跑，不打开游戏窗口。

```bash
python mlogpt/gen.py --run mlogpt-zh-0.72m.msch --ticks 120 --input-text 量持法
```

这条路径需要 Mindustry v160.7 server jar + JRE21，模拟器来自姊妹仓 [mlog-agentskill](https://github.com/EnernalIliary/mlog-agentskill)（`tools/mlogsim.py`）。本仓库不复制它，也不下载 jar——你得自己准备好。

期望结果（实测）：注入「量持法」，8 个 token 的 id 为 `[45,390,400,18,196,45,2,127]`，与 numpy 参考逐位一致，message 上显示「同任意只相同（所」。

### 5. 网页外壳（本地按钮式）

```bash
python web/ui.py     # http://127.0.0.1:8771
```

构建与运行都能点按钮完成，省得记一长串参数。运行侧同样需要 jar（和 `--run` 一样）。

---

## 模型规格

| 项 | 值 |
| --- | --- |
| 名称 | `mlogpt-zh-0.72m` |
| 参数量 | 721,152（权重文件 5,771,362 B） |
| 结构 | decoder-only，pre-LN + 残差，单头注意力 |
| 层数 L / 宽度 d / FFN | 4 / 128 / 512（= 4d），ReLU，无 bias |
| 注意力窗口 S | 16 个 token |
| 词表 V | 4096（汉字 + 中文标点，字符级；未命中记 id 0） |
| 嵌入 | 无嵌入表，固定解析式 `X[t][i] = sin(a_i·t + b_i)` |
| 归一化 | LayerNorm，eps = 1e-5 |
| 采样 | temperature = 1.0，单次比较采样（不存概率表） |
| 一次生成 | N = 8 个 token，输出到 message1..8，每 4 个换行 |

逐张量的形状与参数表见 [docs/01-model.md](docs/01-model.md)。结构上最特别的一条：**只有最后一个位置跨层流动**（第 li 层把 `X[0..S-2]` 直接当 context，只有 `h` 是第 S-1 个位置），前向语句数从 O(S·L·d²) 降到 O(L·d²)。

---

## 实测数字

全部来自一次完整构建 + 一次 headless 实跑，没有估算：

| 项目 | 实测值 |
| --- | --- |
| 处理器 | 10,786 台，全部 world processor（`setrate 1000` ⇒ 1000 条/tick/台） |
| memory cell | 170 个 |
| 语句总数 | 8,767,542（其中前向 7,362,930，占 84.0%） |
| 单台最多语句 | 984 行（mlog 硬上限 999） |
| 蓝图大小 | 71.41 MB（`--out mlogpt-zh-0.72m.msch`） |
| 构建耗时 | 约 52 s |
| 8 个 token | 60 tick，实测 42.7 s |
| 每 token | 约 5 s |
| 每 token 同步点 | 7~8 个 |
| 理论算力上限 | 比实测快约 10 倍（即瓶颈不在算力） |

更细的拆解和瓶颈分析见 [docs/04-performance.md](docs/04-performance.md)。

---

## 诚实的局限

这一节比上面所有数字都重要。

**1. 它不是能聊天的模型。** 验证集 perplexity 94.8，top-1 准确率 18.5%。均匀分布在 4096 个 token 上是 4096，所以 94.8 确实学到了东西；但 18.5% 的 top-1 意味着五分之四的时候它猜不到下一个字。真实输出是「术语碎片」级别，例如输入「量持法」→ 输出「同任意只相同（所」。

**2. 训练语料只有 24.6 万字。** 从本机文档里抽的中文，字符级清洗后 train 224,714 token / val 21,646 token。这个量级能学会「中文的局部搭配」，学不会语言。

**3. 上下文窗口只有 16 个 token。** 它看不到更早的东西，也没有任何持久状态——每一轮都是全新的。

**4. 游戏内手打验收没做。** 目前验证的是**脚本注入**：往 `message11` 里注入文本，检查生成的 token id 序列与 numpy 参考逐位一致。人手在游戏里敲字这条路径没有实测过（输入触发链、0.3 s 稳定等待、清空重来都在注入条件下测试）。

**5. 速度不适合交互。** 约 5 s 一个字，8 个字 42.7 s。你打完一句，它也答不完。

**6. 构建产物很大。** 71.41 MB 的蓝图，粘贴和加载都不轻；构建还要约 52 s。

**7. headless 跑不是开箱即用。** 需要 v160.7 server jar + JRE21，执行器在姊妹仓里，本仓库不打包。

如果你要的是一个能用的中文模型，这个仓库不是；如果你要的是「一个 Transformer 到底能不能塞进 mlog」的答案，它是。

---

## 常见误解与反例

**误解：既然 tiny 配置和 numpy 对上了，全尺寸也一定对吧。**
反例：tiny 用的是完全不同的形状（d=8 / S=8 / L=2 / V=32），权重里还带嵌入表；它只能证明切块与同步逻辑写对了，证明不了 4096 类输出层、16 位置窗口、解析式嵌入在全尺寸下也对。所以全尺寸单独做过一次对拍：注入「量持法」→ `[45,390,400,18,196,45,2,127]`，与参考一致。两件事必须分别验。

**误解：10,786 台处理器、1000 条/tick，算力肯定够，慢是因为游戏优化差。**
反例：8 个字用了 60 tick，而每个 token 的关键路径上有 7~8 个同步点——也就是每 tick 差不多只前进一个同步点。语句执行不是关键路径，barrier 等待才是。理论算力上限比实测快约 10 倍，这 10 倍是被同步吃掉的。

**误解：71 MB 的蓝图 = 模型很大。**
反例：模型只有 721,152 个参数（裸 float32 约 2.9 MB）。71 MB 是**指令流的体积**：mlog 没有数组和循环展开之外的手段，一次 128×512 的矩阵乘要拆成逐元素的读-算-写，语句数上去了，文件就大了。

**误解：ppl 94.8 已经很接近能用了。**
反例：ppl 94.8 意味着平均每次要在 4096 个候选里分摊 94.8 份概率质量（均匀分布是 4096 份，折合约 4.55 nat/字符），听起来比乱猜好得多；但同一组权重下 top-1 只有 18.5%。ppl 和「能顺着你的意思写下去」不是一回事：前者只说明概率分布比均匀集中，后者要求它把大部分质量压在**那一个**正确的字上。

---

## 目录结构

```
README.md                     # 你正在看的这一份
LICENSE                       # MIT
requirements.txt              # numpy>=1.24

docs/
  01-model.md                 # 结构公式、参数表、为什么必须这么简化
  02-train.md                 # 语料、超参、指标、如何用自己的语料重训
  03-inference-in-mlog.md     # numpy → mlog 的映射：数值、切块、同步、分词器
  04-performance.md           # 实测表、瓶颈分析、提速方向
  05-limits.md                # 五条硬限制
  REPRODUCE.md                # 从零复现的完整步骤

weights/
  mlogpt-zh-0.72m-fp32.npz    # 5.5 MB，float32 权重，键：Wq Wk Wv Wo W1 W2 g b Wout
  vocab-zh-4096.json          # 4096 字词表（与训练时一致）
  toy-d8s8l2v32.npz           # 玩具权重，仅用于管线自检
  toy-vocab-32.json

mlogpt/
  gen.py                      # 蓝图生成器：权重 → mlog 语句 → .msch
  reference.py                # numpy 参考前向 + 采样（对拍用）
  msch.py  mschgen.py         # 蓝图写出（vendored，来自 mlog-agentskill，MIT）
  rig.py   rig_chain.py       # 切分与跨块重写规则（vendored）

train/
  train.py                    # numpy 手写反向 + Adam，12000 步
  build_corpus.py             # 语料清洗与词表构建

web/
  ui.py  ui.html              # 本地网页外壳（构建 / 运行，按钮式）
```

`mlogpt/msch.py`、`mlogpt/mschgen.py`、`mlogpt/rig.py`、`mlogpt/rig_chain.py` 是从姊妹仓 vendored 过来的（同样 MIT），保留原样便于对上游同步。

---

## 相关仓库

- **[mlog-agentskill](https://github.com/EnernalIliary/mlog-agentskill)** —— mlog 工具链与 headless 执行器（`tools/mlogsim.py`）在这里。想跑 `--run` 或 `web/ui.py` 的运行侧，去那边拿。本仓库只 vendored 了蓝图写出与切分那四个文件。

---

## 许可

MIT。作者 EnernalIliary（iliary@outlook.com）。

vendored 的 `mlogpt/msch.py`、`mlogpt/mschgen.py`、`mlogpt/rig.py`、`mlogpt/rig_chain.py` 同样以 MIT 分发，来源为 [mlog-agentskill](https://github.com/EnernalIliary/mlog-agentskill)。

---

## 引用

```bibtex
@misc{mlogpt2026,
  author = {EnernalIliary},
  title  = {mlogpt: a 0.72M-parameter Chinese Transformer running in Mindustry mlog},
  year   = {2026},
  url    = {https://github.com/EnernalIliary/mlogpt}
}
```

