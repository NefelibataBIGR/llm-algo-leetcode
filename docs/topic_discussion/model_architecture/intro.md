# 大模型结构和原理专题

## 专题概览

本专题把 `Part 02` 的 `01-08` 串成一条结构理解路线，回答“LLaMA 类模型的核心组件是什么，它们在 block 里怎么连接，张量形状怎么流动”。

`01-08` 已经分别讲了 RMSNorm、SwiGLU、RoPE、Attention、LLaMA Block、MoE 和架构技巧；本专题负责把这些单页重新放回同一张结构地图里，降低初学者读单个组件时的抽象感。

这也是一个样板横向专题，用来验证专题页不只是文件索引，而是可以同时承载叙事、文献和图。

## 建设标准

横向专题至少要同时具备三层内容：

- `文字串联`：用问题驱动的叙事，把主题的重要性、技术演化和跨 Part 关系讲清楚。
- `文献锚点`：用 3 到 5 篇代表性论文或官方文档做溯源锚点，回答“这篇为什么值得读”。
- `可视化资产`：用一张图把专题里的核心关系画出来，重点是降低抽象感，而不是追求美观。

如果一个专题暂时还做不到三层齐备，至少要先把 `文字串联` 做起来，再逐步补文献和图。

## 内容映射

### 来源与前置

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1A / 1B` | 结构理解前需要的参数量、规模估算和访存直觉 |
| `Part 2.1` | RMSNorm、SwiGLU、RoPE、Attention |
| `Part 2.2` | LLaMA Block、MoE Router、Load Balancing、Architecture Tricks |
| `Part 2.3` | 训练微调前需要知道哪些层会被 LoRA 或优化器触达 |

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

- 如果你觉得 `01-04` 抽象，先跳到 `05` 看完整 block 图，再回头补组件。
- 如果你正在学 LoRA，至少要看懂 `04` 里的 attention projection 和 `02` 里的 MLP projection。
- 如果你只想先完成训练微调主线，`06 / 07` 可以作为 MoE 扩展选读。
- 如果你要复制这个专题模板，优先保留 `文字串联 / 文献锚点 / 可视化资产` 三段。

## 正文页

- [casebook.md](./casebook.md)：按“组件职责 / 输入输出形状 / 常见误区 / 检查清单”展开。
- [walkthrough.md](./walkthrough.md)：按一条 token hidden state 连续走完整 block 的方式展开。

## 相关专题

- [训练微调闭环专题](../fine_tuning_training/intro.md)：当你想把这些结构接到 SFT、LoRA 和项目报告时看这里。
- [推理优化专题](../inference_optimization/intro.md)：当你关心 Attention、KV cache 和生成速度时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当你关心结构带来的显存和性能压力时看这里。

## 状态

当前为专题入口页，且作为横向专题样板优先建设。`01-08` 的源 notebook 已完成第一轮轻量可视化，后续可继续补正文案例、连续 walkthrough 和更完整的结构图。
