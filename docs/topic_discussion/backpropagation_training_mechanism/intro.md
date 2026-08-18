# 反向传播与训练机制专题

## 专题定位

本专题用于串起训练机制主线：先看梯度怎么沿计算图回传，再看 attention backward、loss 对齐、activation 保存、checkpointing、offload 和梯度累积怎样一起影响训练节奏与显存代价。这里聚焦训练侧基础机制；如果问题已经进入项目交付或训练路线选择，应转到监督微调专题。

## 主学习线

`Task1-5` 是学习路线，指向 `Part00 / Part01 / Part02` 的具体小节；最后一列主要对应 `01-05` 的专题正文页，`06` 是图册补充。

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
