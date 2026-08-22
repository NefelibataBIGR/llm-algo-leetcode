# 反向传播与训练机制专题

## 专题定位与 Infra 层定位

本专题串起训练机制主线：先看梯度怎么沿计算图回传，再看 attention backward、loss 对齐、activation 保存、checkpointing、offload 和梯度累积怎样一起影响训练节奏与显存代价。它主要连接 Infra-L2–Infra-L3：Infra-L2 解释算子、kernel 和自动求导如何执行，Infra-L3 解释框架如何构建计算图、保存 activation 并调度 backward；Infra-L1 的显存容量与带宽是边界，checkpoint、offload 和梯度累积是跨层策略。

因此，学习者需要同时看计算、内存和通信代价，而不是把本专题当作独立优化方案。若问题进入 SFT、LoRA 或训练项目交付，应转到监督微调专题；若重点是显存预算和策略选型，应转到显存优化专题。

## 推荐入口

推荐把本专题作为 [监督微调专题](../fine_tuning_training/intro.md) 或 [显存优化专题](../memory_performance_tuning/intro.md) 的机制桥接。需要理解训练为什么变慢、爆显存或必须做 checkpointing 时，再进入对应的 Task，而不是把本专题当作独立项目线顺序完成。

## 前置阅读

建议先具备 `Part 00` 的 PyTorch 张量与自动求导基础；如果要进入 attention backward、checkpointing 或 offload，可直接按表中 `17 -> 18 -> 19 -> 42` 回看来源 notebook。

## 主学习线

`Task1-5` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列主要对应 `01-05` 的专题正文页，`06` 是图册补充。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | backward 总览与计算图 | `17` | [01 Backpropagation and Graph](./01_backpropagation_and_graph.md) |
| Task2 | autograd 与 attention backward | `17 -> 18` | [02 Autograd and Attention Backward](./02_autograd_and_attention_backward.md) |
| Task3 | loss 对齐与显存账本 | `18` | [03 Loss Alignment and Memory Ledger](./03_loss_alignment_memory_ledger.md) |
| Task4 | checkpointing 与 offload | `19 -> 42` | [04 Checkpointing and Offload](./04_checkpointing_and_offload.md) |
| Task5 | 梯度累积、训练闭环与 profiling | `12 -> 73 -> 74` | [05 Accumulation, Decision and Profiling](./05_accumulation_decision_profiling.md) |

## 正文与跳转

先按上面的 `Task1-5` 走 notebook 主线；遇到“梯度到底怎么回去”“为什么 activation 要保存”“checkpointing 和 offload 本质差别是什么”时，再回来看对应的专题正文。想看汇总版就进 [反向传播与训练机制正文](./casebook.md)，想按连续故事线走一遍就进 [反向传播与训练机制深入阅读](./walkthrough.md)。工具层补充放在 [训练工具桥](./training_tooling_bridge.md)，图册补充放在 [06 Visual Assets](./06_visual_assets.md)。

如果问题已经跨到别的专题：
[监督微调专题](../fine_tuning_training/intro.md) 负责训练闭环与项目交付，[显存优化专题](../memory_performance_tuning/intro.md) 负责训练侧显存 trade-off，[Profiling 专题](../profiling/intro.md) 负责证据链与热点定位。

## 环境与验证

计算图、loss 和小规模 backward 实验通常可用 CPU；长序列、checkpoint/offload 和真实训练性能比较建议使用 GPU。验证时应区分数值正确性、峰值显存和 step time，不能把单项指标改善直接当成整体优化结论。
