# 大模型结构和原理专题

## 专题概览

本专题不再只把 `Part 02` 的 `01-08` 串成一条结构理解路线，而是把“大模型结构和原理”升级成一门**结构演进史 + 代表模型剖析**专题。

它要回答的不是“LLaMA 类模型的核心组件是什么”这么单一的问题，而是下面这一组问题：

- 为什么 BPE、Embedding、Norm、Attention、RoPE 会不断演化
- 每一代方案分别解决了什么矛盾
- 这些变化如何影响 block 组装、训练、推理和显存
- LLaMA、DeepSeek、Qwen、Mistral、Gemma 这类代表模型到底选了什么结构

`01-08` 仍然是专题的基础骨架，但它们只是起点，不是全部。后续专题正文会按“模块演进 + 代表模型”两条线展开，把单页重新放回同一张知识地图里。

## 建设标准

横向专题至少要同时具备三层内容：

- `文字串联`：用问题驱动的叙事，把主题的重要性、技术演化和跨 Part 关系讲清楚。
- `文献锚点`：用 3 到 5 篇代表性论文或官方文档做溯源锚点，回答“这篇为什么值得读”。
- `可视化资产`：用一张图把专题里的核心关系画出来，重点是降低抽象感，而不是追求美观。

如果一个专题暂时还做不到三层齐备，至少要先把 `文字串联` 做起来，再逐步补文献和图。

## 专题分块

这个专题后续会按下面几个小节扩展，不再只是入口页：

1. `Tokenization / BPE / Embedding 演进`
2. `归一化发展历程`
3. `Attention 发展历程`
4. `RoPE 与位置编码发展历程`
5. `Block 组装与残差路径`
6. `代表模型专题`
7. `跨模块对照表`
8. `可视化资产`

每一块都应该可以单独成节，便于继续补充：

- 经典论文：回答“这个方向最早解决了什么问题”
- 代表实现：回答“现在常见模型怎么落地”
- 前沿论文：回答“这个方向现在往哪里演化”

### 建议文件设计

为了让每个主题都能独立成页，建议后续按下面的文件拆分：

- `tokenization_embedding.md`
- `norm_evolution.md`
- `attention_evolution.md`
- `rope_position_encoding.md`
- `block_residual_path.md`
- `representative_models.md`
- `cross_module_comparison.md`
- `visual_assets.md`

其中：

- `tokenization_embedding.md` 讲 BPE、词表、Embedding 与表示层的关系
- `norm_evolution.md` 讲 LayerNorm、Pre-Norm、RMSNorm 及其演化
- `attention_evolution.md` 讲 MHA、MQA、GQA、稀疏 attention 及其取舍
- `rope_position_encoding.md` 讲 RoPE 与长上下文扩展
- `block_residual_path.md` 讲 block 组装、残差路径和信息流
- `representative_models.md` 讲 LLaMA、DeepSeek、Qwen、Mistral、Gemma 等模型的结构选择
- `cross_module_comparison.md` 负责把各模块的演进关系横向拉通
- `visual_assets.md` 负责沉淀 block 图、演进时间线和对照图

### 每页固定模板

每个独立主题页建议统一使用下面的结构，便于后续扩展论文和案例：

1. `问题起点`
2. `演化过程`
3. `代表模型`
4. `经典论文`
5. `前沿论文`
6. `与 Part 02 01-08 的对应关系`
7. `可视化提示`

这样每页都能独立阅读，也能放回整套专题体系里串起来读。

### 已落地主题页

当前已经先落地了下面 10 页：

- [tokenization_embedding.md](./tokenization_embedding.md)
- [norm_evolution.md](./norm_evolution.md)
- [attention_evolution.md](./attention_evolution.md)
- [rope_position_encoding.md](./rope_position_encoding.md)
- [transformer_decoder.md](./transformer_decoder.md)
- [mlp_ffn_evolution.md](./mlp_ffn_evolution.md)
- [block_residual_path.md](./block_residual_path.md)
- [representative_models.md](./representative_models.md)
- [cross_module_comparison.md](./cross_module_comparison.md)
- [visual_assets.md](./visual_assets.md)

## 叙事骨架

这个专题不是按文件顺序读，而是按问题顺序读。

- 问题导入：为什么结构要单独成专题，因为 tokenization、norm、attention、位置编码和 block 组装决定了后续训练、推理、显存和微调的共同边界。
- 核心脉络：每一代大模型结构的变化，本质上都是在表达能力、稳定性、上下文建模和计算代价之间重新做平衡。
- 技术演进：先讲模块怎么演化，再讲这些模块如何被组装成 block，最后看代表模型为什么会选某一条路线。
- 关键取舍：BPE 怎么影响词表和 embedding，norm 为什么从 LayerNorm 走到 RMSNorm，attention 为什么从 MHA 走到 MQA/GQA，RoPE 为什么会成为默认位置编码，DeepSeek 这类模型为什么会继续改 attention 结构。

## 内容映射

### 来源与前置

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1A / 1B` | 结构理解前需要的参数量、规模估算和访存直觉 |
| `Part 2.1` | RMSNorm、SwiGLU、RoPE、Attention |
| `Part 2.2` | LLaMA Block、MoE Router、Load Balancing、Architecture Tricks |
| `Part 2.3` | 训练微调前需要知道哪些层会被 LoRA 或优化器触达 |
| `后续扩展页` | BPE、Embedding、Norm 演进、Attention 演进、RoPE 演进、代表模型专题 |

### 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `01` | RMSNorm 在 block 中的位置与归一化方向 | [01 RMSNorm Tutorial](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb) |
| `02` | SwiGLU 的 gate / up / down 三条分支 | [02 SwiGLU Activation](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb) |
| `03` | RoPE 如何作用在 Query / Key 上 | [03 RoPE Tutorial](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb) |
| `04` | MHA / GQA / MQA 的 Q/K/V head 关系 | [04 Attention MHA GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) |
| `05` | 一个 LLaMA Block 如何串起 `01-04` | [05 LLaMA3 Block Tutorial](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) |
| `06` | token 如何被 Router 分配给 Top-K experts | [06 MoE Router](../../02_PyTorch_Algorithms/06_MoE_Router.ipynb) |
| `07` | 为什么 MoE 需要负载均衡损失 | [07 MoE Load Balancing Loss](../../02_PyTorch_Algorithms/07_MoE_Load_Balancing_Loss.ipynb) |
| `08` | Weight tying、Gemma RMSNorm 等结构技巧放在哪里 | [08 Architecture Tricks](../../02_PyTorch_Algorithms/08_Architecture_Tricks.ipynb) |

### 未来小节

后续正文会继续补下面这些专题页：

- `BPE / Embedding`：解释 tokenization 和词表如何影响表示层
- `Norm`：讲 LayerNorm、Pre-Norm、RMSNorm 的演化
- `Attention`：讲 MHA、MQA、GQA、稀疏 attention 的演化
- `RoPE`：讲旋转位置编码及其长上下文扩展
- `Representative Models`：讲 LLaMA、DeepSeek、Qwen、Mistral、Gemma 等模型的结构选择

### 叙事主线

这个专题建议按“一个 token 的 hidden state 如何穿过 LLaMA block”来读。

- 先回答为什么要单独看结构：结构决定 hidden state 怎么流动、参数怎么分配、信息怎么在层间传递。
- 再回答核心矛盾是什么：dense block 需要在稳定性、表达能力和计算成本之间找平衡。
- 最后回答各文件分别解决什么环节：`01-04` 解决基础组件，`05` 解决 block 组装，`06-07` 解决 MoE，`08` 解决真实实现和教科书结构的差异。

### 文献锚点

| 文献 | 为什么值得读 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 把 RMSNorm、RoPE、SwiGLU 和 block 结构放在同一实现框架里的代表性入口。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 解释为什么 RMSNorm 能成为大模型里常见的 norm 选择。 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 解释 RoPE 为什么会成为现代 LLM 的默认位置编码之一。 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 解释为什么 SwiGLU 这类门控 MLP 变体值得进入 Transformer FFN。 |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | 解释 MoE 为什么要把 router、expert 和负载均衡拆开看。 |

### 可视化资产

建议补一张完整 block 图，至少标出以下路径：

- `token embedding -> RMSNorm -> Attention -> Residual`
- `Residual -> RMSNorm -> SwiGLU / MLP -> Residual`
- `Residual -> optional MoE / tricks`

图的目标不是美观，而是让人一眼知道组件在 block 里的位置关系。

## 阅读方式

- 如果你想先建立总图，先看“专题分块”和“叙事骨架”。
- 如果你想沿着现有实现理解，先看 `01-08`，再回到未来小节。
- 如果你想深挖某个模块，优先看对应的演进史，再看代表模型怎么落地。
- 如果你要复制这个专题模板，优先保留 `文字串联 / 文献锚点 / 可视化资产` 三段，再补模块演进和代表模型。

## 正文页

- [casebook.md](./casebook.md)：当前作为“结构地图 + 模块演进 + 代表模型对照”的正文骨架。
- [walkthrough.md](./walkthrough.md)：当前作为“连续故事线 + 结构演化 + 真实实现对照”的正文骨架。

## 相关专题

- [训练微调闭环专题](../fine_tuning_training/intro.md)：当你想把这些结构接到 SFT、LoRA 和项目报告时看这里。
- [推理优化专题](../inference_optimization/intro.md)：当你关心 Attention、KV cache 和生成速度时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当你关心结构带来的显存和性能压力时看这里。

## 状态

当前为专题入口页，且作为横向专题样板优先建设。`01-08` 的源 notebook 已完成第一轮轻量可视化，后续要补的是模块演进史、代表模型专题和更完整的可视化资产。
