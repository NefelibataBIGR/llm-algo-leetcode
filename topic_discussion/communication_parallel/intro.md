# 通信与并行专题

## 专题定位

本专题用于串起多卡并行主线：先看通信原语和拓扑，再看 DDP、FSDP、ZeRO、Pipeline、Tensor Parallel 和专家并行分别切了什么，最后把通信热点和 benchmark 收回并行选型结论。这里重点关注切分层级与通信代价；如果问题先表现为单机推理速度，应转到推理优化专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part01 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 通信拓扑与 AllReduce 前置 | `P1:1C -> 05 -> 20` | [01 Why Parallel and Communication](./01_why_parallel_and_communication.md) |
| Task2 | DDP 到 FSDP / ZeRO 的状态分摊 | `06 -> 27` | [02 Data Parallel and Synchronization](./02_data_parallel_and_synchronization.md) |
| Task3 | Pipeline Parallel 的时序与气泡 | `28` | [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md) |
| Task4 | Tensor Parallel 的切分与代价 | `29` | [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md) |
| Task5 | 通信 profiling 与热点定位 | `46 -> 79` | [05 Expert Parallel and Communication Hotspots](./05_expert_parallel_and_communication_hotspots.md) |
| Task6 | 并行项目收口 | `80 -> 81` | [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“为什么多卡不一定更快”“不同切分到底换来了什么”时，再回来看对应的专题正文。想看汇总版就进 [通信与并行正文](./casebook.md)，想按连续故事线走一遍就进 [通信与并行深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责证据链与等待热点，[显存优化专题](../memory_performance_tuning/intro.md) 负责显存分摊 trade-off，[监督微调专题](../fine_tuning_training/intro.md) 负责训练工程闭环，[编译与图优化专题](../compiler_graph_optimization/intro.md) 负责执行模型与 backend 约束。
