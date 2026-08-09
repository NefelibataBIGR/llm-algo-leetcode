# 通信与并行深入阅读

## 主故事线

如果把这条线写完整，通常会经历这样的过程：先是“单卡不够装了”，于是考虑加卡；一加卡之后显存确实下来了，但 step time 没按预期缩短，于是先沿着 `05 -> 20` 看同步原语和拓扑关系，再沿着 `06 -> 27 -> 28 -> 29` 判断到底该切状态、切层还是切算子；当训练还是没快时，再用 `46 -> 79 -> 66` 看通信等待、bubble 和 benchmark 结果，确认并行策略到底是把收益做出来了，还是只是把瓶颈从显存转成了通信；最后回到 `2.8`，把策略和训练目标一起重新选一遍。

这条故事本身也是从已有主线长出来的：

- `Part01` 先把拓扑、NCCL、AllReduce 和显存共享立住；
- `Part02` 再把 ZeRO、Pipeline、Tensor Parallel、MoE 和 benchmark 落成实现；
- 横向专题负责把它们串成“为什么切、怎么切、代价去哪了”的连续叙事。

## 端到端案例

一个更完整的并行选型过程，通常是从“单卡放不下、加卡又没快多少”开始的。先沿着 `05 -> 20` 看通信原语和拓扑，确认多卡同步的基本语义；再沿着 `06 -> 27 -> 28 -> 29` 看到底是状态切分、层切分还是算子切分更合适；当 step time 还是没有明显下降时，再用 `46 -> 79 -> 66` 检查通信等待、bubble 和 benchmark 结果，确认问题是不是被从显存转移到了通信；最后回到 `2.8`，把并行策略和训练目标重新对齐。

如果你已经知道自己的问题落在哪一层，可以直接跳到对应编号页：

- [01 Why Parallel and Communication](./01_why_parallel_and_communication.md)
- [02 Data Parallel and Synchronization](./02_data_parallel_and_synchronization.md)
- [03 State Sharding and ZeRO](./03_state_sharding_and_zero.md)
- [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md)
- [05 Expert Parallel and Communication Hotspots](./05_expert_parallel_and_communication_hotspots.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)

## 阅读建议

- 先把长故事线读完，再去看正文里的案例和清单。
- 如果你要做并行选型，建议把这里和 [通信与并行正文](./casebook.md) 一起看。
- 如果你要先看诊断方法，可以先转到 [Profiling 专题](../profiling/intro.md)。
