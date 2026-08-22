# 反向传播与训练机制正文

这页只做训练机制问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先确认计算图和梯度路径，再检查 loss 对齐与显存账本；只有明确压力来源后，才在 checkpoint、offload、梯度累积或 profiling 之间选择。不要把所有 OOM 都归因于 batch size。

## 判断表

先分清问题在计算图、autograd、loss 对齐、activation 保存还是训练节奏，再判断它是不是已经转成显存或 profiling 问题。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 会写 forward，但解释不清梯度怎么回去 | `graph mismatch` | [01](./01_backpropagation_and_graph.md) | 先画清计算图和梯度路径 |
| attention 能跑，但 backward 看不懂 | `autograd / attention backward` | [02](./02_autograd_and_attention_backward.md) | 看 `grad_fn`、`saved_tensors`、`dV -> dP -> dS -> dQ/dK` |
| loss 在降，但监督口径不可信 | `loss alignment mismatch` | [03](./03_loss_alignment_memory_ledger.md) | 检查 `mask / shift / ignore_index / labels` |
| 训练侧显存明显过高 | `activation residency` | [04](./04_checkpointing_and_offload.md) | 区分 checkpointing 和 offload 的代价模型 |
| 训练能跑，但 step 口径混乱 | `training rhythm mismatch` | [05](./05_accumulation_decision_profiling.md) | 检查 accumulation、optimizer step、effective batch |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 计算图 | 梯度沿什么路径回传 | 会用 API 就等于懂 backward |
| `saved_tensors` | 哪些状态必须保留 | 把公式理解和保存代价分开看 |
| 标签对齐 | supervision 是否真的进了 loss | loss 有值就算标签正确 |
| activation 保存 | 显存主峰值是不是来自中间状态 | 把所有问题都归到 batch 太大 |
| accumulation | backward 次数和 step 次数是否一致 | 训练能跑就等于训练节奏对了 |

## 本节要点

这页的职责不是再讲一遍 backward 流程，而是把训练机制里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小机制审计模板

记录 `梯度路径 -> 监督边界 -> 显存组成 -> 调度策略 -> step/quality 对照 -> 下一步`。如果只证明显存下降，没有说明 loss、吞吐或有效 batch 是否保持，结论仍应标记为待验证。
