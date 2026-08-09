# 反向传播与训练机制深入阅读

## 主故事线

这条专题的阅读顺序可以理解为：

`graph -> autograd -> attention backward -> loss / memory ledger -> checkpointing / offload -> accumulation -> profiling`

核心不是把公式背完，而是把“梯度怎么回去、哪些状态要留、为什么 backward 会变慢、怎么把这些问题放回训练闭环”讲清楚。

## 01 反向传播总览与计算图

先把问题框住：

- backward 为什么会影响训练能不能跑起来
- 计算图如何决定梯度路径
- 为什么有些状态必须留，有些可以重算

## 02 Autograd 与 Attention Backward

先把实现层的关键接口和 attention 链路看懂：

- `grad_fn` 和 `saved_tensors` 是什么
- `autograd.Function` 为什么是最小的 backward 入口
- attention 的 `dV -> dP -> dS -> dQ / dK` 怎么走

## 03 Loss Backward、标签对齐与显存账本

再把监督口径和显存代价放在一起看：

- `mask / shift / ignore_index` 如何工作
- prompt / response 为什么不能一视同仁
- activation、参数、梯度和 optimizer state 各占什么位置

## 04 Checkpointing 与 Offload

再看两类最重要的显存优化：

- checkpointing 是重算
- offload 是搬运
- 两者分别在优化什么、代价是什么

## 05 梯度累积、训练闭环与 Profiling

最后把 backward 放回训练节奏里：

- micro-batch 如何合成 effective batch
- backward / step / profiling 怎么配合
- 怎么判断某个优化是真的有效

## 阅读建议

1. 先从 `01` 开始，建立 backward 的基本框架。
2. 再看 `02`，把实现接口和 attention 反向链路对上。
3. 再看 `03`，把监督口径和显存代价对齐。
4. 再看 `04`，理解 backward 的两类核心省显存方法。
5. 最后看 `05`，把这些机制放回优化决策和 profiling 闭环。
