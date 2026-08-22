# 03. Checkpointing and Offload | Checkpointing 与 Offload

## 页面目标

这一页回答的是：为什么 checkpointing 和 offload 会成为训练显存优化主线，以及它们分别在拿什么换空间。

## 问题起点

当 activation 成为训练峰值主因后，系统只剩两条主路：

- 少存一些，回头再算；
- 先存着，但搬到别的存储层。

这正是 checkpointing 和 offload 的本质区别。它们都在省 activation，却通过完全不同的代价模型完成。

## 你要先确认什么

- activation 是否已经是训练峰值主因。
- 当前系统更能接受重算，还是更能接受搬运。
- 带宽、PCIe / NVLink 路径是否足以支撑 offload。

## 核心矛盾

`checkpointing` 的矛盾是“少留状态，但后向时要多做一次前向片段重算”；`offload` 的矛盾是“状态仍然存在，但需要跨存储层搬运并等待返回”。两者都不是免费收益，差别只在于你把代价付给计算还是传输。

## 演化路径

1. 先识别 activation 是否值得优化。
2. 用 checkpointing 把一部分状态从“存储”改成“重算”。
3. 用 offload 把状态从 GPU 驻留改成外部存储层驻留。
4. 再用 profiling 看重算和搬运是否把时间赔过头。

## 关键取舍

- `checkpointing` 更适合计算相对便宜、重算可接受的片段。
- `offload` 更适合显存太紧但外部带宽还可承受的环境。
- 两者可以组合，但组合后更需要 benchmark，不然很容易只看到显存下降，看不到训练时间恶化。

![Checkpointing and offload trade-off](/topic_discussion/memory_performance_tuning/checkpointing_offload.svg)

## 文献锚点

- activation checkpointing 经典论文：理解时间换空间的基本模式。
- activation offload / memory hierarchy 相关资料：理解搬运路径为什么常成为隐藏成本。

## 对应 Part 02

- `19` Activation Checkpointing and Activation Offload
- `42` Activation Offload
- `74` Profiling Driven End-to-End Optimization

## 典型阅读入口

- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 本节要点

checkpointing 和 offload 都是在省 activation，但一个主要赔计算，一个主要赔传输。
