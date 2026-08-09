# 反向传播与训练机制专题

## 专题概览

本专题把 `Part 02` 里分散的反向传播、梯度流、训练调度和显存代价串成一条基础横向线，回答四个问题：

- 梯度是怎么回去的
- backward 里哪些张量必须保留
- 梯度累积如何改变训练节奏
- checkpointing / offload 为什么会改变显存和性能

它不是公式合集，而是训练机制的认知路线图。阅读时的目标是先建立 backward 的统一心智模型，再把它放回训练闭环里验证。

## 章节安排

推荐按下面的 `01-05` 顺序阅读。原先的十个拆分视角，已经合并成更厚的五个机制页：

1. `反向传播总览与计算图`
   - backward 专题的目标、范围、链式法则和调图视角
2. `Autograd 与 Attention Backward`
   - `grad_fn`、`saved_tensors`、自定义 backward，以及 attention 的反向链路
3. `Loss Backward、标签对齐与显存账本`
   - `mask / shift / ignore_index`、监督口径、激活保存和显存账本
4. `Checkpointing 与 Offload`
   - `重算换显存` 与 `搬运换显存` 的边界、代价和适用场景
5. `梯度累积、训练闭环与 Profiling`
   - `micro-batch / effective batch`、训练调度、瓶颈判断和收益验证

## 职责边界

这个专题只负责反向传播和训练机制的基础认知，不负责训练项目收口、不负责推理优化主线，也不负责完整的显存总论。

- `反向传播总览与计算图` 负责建立梯度回传的共同心智模型。
- `Autograd 与 Attention Backward` 负责把图上的梯度路径落到 PyTorch 和 attention 算子上。
- `Loss Backward、标签对齐与显存账本` 负责把监督口径和显存代价对齐。
- `Checkpointing 与 Offload` 负责解释两类最重要的显存优化。
- `梯度累积、训练闭环与 Profiling` 负责把 backward 放回真实训练和验证流程里。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1B / 1D` | 反向传播前需要理解的 GPU 访存、执行模型和调度边界 |
| `Part 2.0` | 最小 autograd、backward 热身和梯度流直觉 |
| `Part 2.5` | Attention backward、activation backward、loss backward |
| `Part 2.6 / 2.5` | 梯度累积和训练循环里的 backward 调度 |
| `Part 2.5 / 2.7B` | checkpointing / offload 的显存代价 |
| `Part 2.9` | backward 热点、训练性能分析和收益验证 |

## 文献锚点

这一专题的论文锚点不是为了堆引用，而是为了给每个机制页一个“从哪里来、为什么会变成这样”的起点。

| 锚点 | 覆盖页面 | 为什么值得先看 |
|:---|:---|:---|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | `01 / 03` | 反向传播的起点，先理解梯度信号如何穿过多层网络。 |
| [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) | `01 / 02` | 看 automatic differentiation 如何把链式法则变成可执行的图。 |
| [Automatic Differentiation in ML: Where we are and where we should be going](https://arxiv.org/abs/1810.11530) | `02` | 补充 AD 的工程化视角，适合理解 graph-based autograd。 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | `02 / 03` | attention 和 causal loss 的共同起点。 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | `03 / 04` | checkpointing 的经典起点，直接对应“重算换显存”。 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | `04 / 05` | 先建立 offload / 分片 / memory hierarchy 的系统视角。 |
| [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) | `04` | 进一步理解 offload 的 CPU/GPU 搬运代价。 |
| [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) | `04 / 05` | 代表 heterogeneous memory 的更激进路线。 |
| [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) | `05` | 解释为什么 batch / step / generalization 会纠缠在一起。 |
| [Enabling Large Batch Size Training for DNN Models Beyond the Memory Limit While Maintaining Performance](https://arxiv.org/abs/2110.12484) | `05` | 梯度累积 / micro-batch 的直接工程背景。 |

## 推荐入口

- 如果你还没看过反向传播的基础，先从 `01` 开始。
- 如果你想把 attention 的反向链路看明白，接着看 `02`。
- 如果你关心监督信号怎么进 loss 和显存，接着看 `03`。
- 如果你关心 backward 为什么吃显存，接着看 `04`。
- 如果你关心训练节奏和闭环验证，最后看 `05`。

## 正文页

- [casebook.md](./casebook.md)：按五个机制块展开“问题-误区-验证”。
- [walkthrough.md](./walkthrough.md)：按一条训练样本的 backward 故事线展开，直到 profiling 复盘。

## 章节页

- [01. 反向传播总览与计算图](./01_backpropagation_and_graph.md)
- [02. Autograd 与 Attention Backward](./02_autograd_and_attention_backward.md)
- [03. Loss Backward、标签对齐与显存账本](./03_loss_alignment_memory_ledger.md)
- [04. Checkpointing 与 Offload](./04_checkpointing_and_offload.md)
- [05. 梯度累积、训练闭环与 Profiling](./05_accumulation_decision_profiling.md)

## 相关专题

- [训练微调闭环专题](../fine_tuning_training/intro.md)：当你要把 backward 放进 SFT / LoRA 训练闭环时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当 backward 牵涉 checkpointing / offload / 显存账本时先看这里。
- [Profiling 专题](../profiling/intro.md)：当你需要证明 backward 的瓶颈和收益时先看这里。

## 专题状态

当前为专题入口页，后续会逐步补成和 `model_architecture` 一样的三层结构，但它的主轴是机制链路，而不是组件演化。
