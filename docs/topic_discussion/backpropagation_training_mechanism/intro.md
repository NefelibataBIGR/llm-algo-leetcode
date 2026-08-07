# 反向传播与训练机制专题

## 专题概览

本专题用于把 `Part 02` 中分散的反向传播、梯度流、训练调度和显存代价串成一条基础横向线，回答“梯度是怎么回去的、训练里 backward 怎么调度、为什么 backward 会牵动显存和性能”。

这条线覆盖 `00 / 17 / 18 / 12 / 19 / 42 / 74`：先从最小 autograd 热身开始，再看 attention / activation / loss 的反向，接着处理梯度累积与 backward 调度，随后理解 checkpointing / offload 这类显存策略，最后用 profiling 把 backward 的收益和代价验证清楚。

## 职责边界

这个专题只负责反向传播和训练机制的基础认知，不负责训练项目收口、不负责推理优化主线，也不负责完整的显存总论。

- `Autograd / Backward` 关注 `grad_fn`、`saved_tensors`、梯度流和最小自定义 backward。
- `Attention / Activation / Loss` 关注反向链路、保留张量和监督区间。
- `Gradient Accumulation` 关注 micro-batch、effective batch 和 backward 调度。
- `Checkpointing / Offload` 关注 backward 的显存代价和时间换空间策略。
- `Profiling` 关注 backward 热点、收益验证和前后对比。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1B / 1D` | 反向传播前需要理解的 GPU 访存、执行模型和调度边界 |
| `Part 2.0` | 最小 autograd、backward 热身和梯度流直觉 |
| `Part 2.5` | Attention backward、activation backward、loss backward |
| `Part 2.6 / 2.5` | 梯度累积和训练循环里的 backward 调度 |
| `Part 2.5 / 2.7B` | checkpointing / offload 的显存代价 |
| `Part 2.9` | backward 热点、训练性能分析和收益验证 |

## Part 1 相关前置

- [1B](../../01_Hardware_Math_and_Systems/1B.md)：先看 GPU 架构、访存和显存路径，知道 backward 为什么会留下很多中间量。
- [1D](../../01_Hardware_Math_and_Systems/1D.md)：先看执行模型和调度边界，理解 backward kernel 为什么会和 stream / execution model 挂钩。

## Task1-6 路线

| Task | 内容 | 章节 |
|:---|:---|:---|
| Task1 | 反向传播基础入口 | [00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb) |
| Task2 | Attention / Activation / Loss 的反向机制 | [17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)、[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb) |
| Task3 | 训练中的 backward 调度 | [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) |
| Task4 | backward 的显存代价 | [19 Activation Checkpointing](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)、[42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) |
| Task5 | backward 的性能观察 | [74 Profiling-Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) |
| Task6 | 机制收口与复盘 | [casebook.md](./casebook.md)、[walkthrough.md](./walkthrough.md) |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `00` | 最小 autograd 热身、手写 backward、梯度流直觉 | [00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb) |
| `17` | Attention backward、`saved_tensors`、`gradcheck` | [17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb) |
| `18` | activation backward、loss backward、监督区间 | [18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb) |
| `12` | micro-batch、effective batch、梯度累积 | [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) |
| `19` | checkpointing 的重算代价 | [19 Activation Checkpointing](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb) |
| `42` | activation offload 的搬运代价 | [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) |
| `74` | backward 热点、训练性能分析和优化闭环 | [74 Profiling-Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) |

## 推荐入口

- 如果你还没看过反向传播的基础，先从 `00 -> 17 -> 18` 开始。
- 如果你想把训练中的 backward 调度看明白，再接 `12`。
- 如果你关心显存和 backward 的关系，接着看 `19 -> 42`。
- 如果你关心 backward 的性能热点和优化验证，最后看 `74`。

## 入口摘要

- 最短反向传播路线：`00 -> 17 -> 18 -> 12 -> 19 -> 42 -> 74`。
- 训练微调辅助路线：`00 -> 17 -> 18 -> 12`。
- 显存优化辅助路线：`17 -> 18 -> 19 -> 42`。
- 性能验证辅助路线：`74`。

## 正文页

- [casebook.md](./casebook.md)：按“常见错误 / 排障清单 / backward 代价 / 调度口径”展开。
- [walkthrough.md](./walkthrough.md)：按一条训练样本的 backward 故事线展开，直到 profiling 复盘。

## 相关专题

- [训练微调闭环专题](../fine_tuning_training/intro.md)：当你要把 backward 放进 SFT / LoRA 训练闭环时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当 backward 牵涉 checkpointing / offload / 显存账本时先看这里。
- [Profiling 专题](../profiling/intro.md)：当你需要证明 backward 的瓶颈和收益时先看这里。

## 读法建议

- `00` 先把最小 autograd 和 backward 热身跑通。
- `17 / 18` 一起看，先理解反向链路，再看损失和激活的回传口径。
- `12` 用来理解 backward 在训练循环中的调度。
- `19 / 42` 用来理解 backward 和显存策略的关系。
- `74` 用来把这些机制放回 profiling 闭环里验证。

## 专题状态

当前为专题入口页，后续将逐步补充更完整的排障清单、反向传播口径对照和连续故事线。
