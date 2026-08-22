# 02. Training Memory Pressure | 训练侧显存压力

## 页面目标

这一页回答的是：训练为什么会 OOM，训练显存通常先被谁吃掉，以及 batch、activation、optimizer state 应该怎么分开看。

## 问题起点

训练显存问题最常见的表象是“一上大 batch 就炸”，但真正把系统拖垮的往往不是 batch 这个单变量，而是：

- activation 在前向和反向之间持续保留；
- gradient accumulation 把有效 batch 放大；
- optimizer state 和梯度常驻；
- 某些层的中间状态在后向阶段集中抬高峰值。

因此，训练显存不能只问“batch 能多大”，而要问“哪个资源对象正在把峰值顶上去”。

## 你要先确认什么

- OOM 出现在 step 一开始，还是中后段逐渐堆高。
- activation 是不是峰值主因。
- 是否已经在用 accumulation，却仍然没有把账本拆清楚。

## 核心矛盾

训练侧的核心矛盾是：模型希望保留足够多的中间状态做反传，但系统又必须把这些状态压进有限显存预算。越大的 effective batch、越长的序列、越深的模型，越会把这个矛盾推到前台。

## 它如何承接 Task0

[17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb) 解释计算图和梯度流，[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb) 进一步说明 loss、logits 和中间激活在反向传播中的生命周期。本页把这些机制转换成显存问题：哪些张量必须保留、哪些张量可以重算、哪些状态只是 optimizer 或 batch 组织带来的常驻成本。

## 演化路径

1. 先从 batch / sequence length 的粗调开始。
2. 再分清 parameters、gradients、optimizer state、activations 谁是主因。
3. 如果 activation 是主因，就继续看 checkpointing 和 offload。
4. 如果 optimizer state 或参数常驻太高，就回到 sharding / ZeRO / 量化路线。
5. 最后把收益放回 `73 / 74` 看时间代价。

## 关键取舍

- 直接缩 batch 最简单，但会改变吞吐和训练节奏。
- `gradient accumulation` 看似省显存，本质是在时间和 step 组织上换空间。
- activation 优化通常能立刻见效，但很少没有时间代价。

![Training memory pressure](/topic_discussion/memory_performance_tuning/training_memory_pressure.svg)

## 文献锚点

- large batch / gradient accumulation 相关资料：理解 effective batch 如何改变显存与优化步节奏。
- activation memory / training system 论文：理解训练峰值为何多在中间状态上。

## 对应 Part 02

- [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- [17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)、[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)、[19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
- [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)

## 典型阅读入口

- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

训练显存问题首先是资源对象问题，其次才是 batch 参数问题。
