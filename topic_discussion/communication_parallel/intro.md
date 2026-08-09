# 通信与并行专题

## 专题概览
本专题不是重新发明并行课程，而是**承接 `Part01-02` 已经存在的多卡扩展路线**，把分散在通信原语、状态切分、层切分、算子切分和 benchmark 里的内容，重组为一条“系统为什么会走向并行，以及通信代价怎么决定并行是否值得”的故事线。

## 职责边界

这个专题只负责多卡训练和推理中的并行策略、通信代价和调度边界，不负责单机推理优化本身，也不负责编译链路。

- `NCCL / AllReduce` 关注最基础的通信原语和同步代价。
- `ZeRO` 关注参数、梯度和优化器状态的切分与显存分摊。
- `Pipeline Parallelism` 关注 micro-batch 的流水线时序和气泡问题。
- `Tensor Parallelism` 关注张量切分后的通信与计算平衡。
- `Communication Profiling` 关注通信热点、等待时间和 overlap。

这个专题不是单纯的并行方法索引，而是一个“切分层级与代价来源”的知识组织轴。

## 承接已有学习路线

这个专题的正文应当建立在已有主线上，而不是脱离主线重新列方法。

### Part01：硬件、拓扑与通信前置

### Part02：并行实现与项目验证

- `27 / 28 / 29 / 46 / 79` 负责把 ZeRO、Pipeline、Tensor Parallel、通信 profiling 和分布式 benchmark 落到实现与验证。

## Part 1 相关前置

- [1C](../../01_Hardware_Math_and_Systems/1C.md)：先看通信拓扑和显存共享，知道多卡为什么一定会带来通信代价。
- [05](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb)：先看通信拓扑和显存共享关系，确认多卡切分前需要看什么。
- [20](../../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb)：先看 NCCL / AllReduce 基础原语，知道通信语义怎么落到实现。

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `1C` | 多卡通信与显存共享的总入口，先看 Group Overview / Asset Overview / Learning Path | [1C 多卡通信与显存共享](../01_Hardware_Math_and_Systems/1C.md) |
| `05` | 通信拓扑和显存共享关系，适合先看页面里的核心职责与判断链路 | [05 Communication Topologies](../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) |
| `06` | 显存开销与 ZeRO 收益估算，适合先看公式与对比结论 | [06 VRAM Calculation and ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) |
| `20` | NCCL 和 AllReduce 基础原语，适合先看原语定义和通信语义 | [20 NCCL and AllReduce Basics](../01_Hardware_Math_and_Systems/20_NCCL_and_AllReduce_Basics.ipynb) |
| `2.8` | 分布式并行策略主线，适合先看 Group Overview / Asset Overview / Learning Path | [2.8 分布式并行策略](../02_PyTorch_Algorithms/2_8.md) |
| `27` | ZeRO 的显存分摊与收益，适合看 Step 1-4 的收益与代价 | [27 ZeRO Optimizer Sim](../02_PyTorch_Algorithms/27_ZeRO_Optimizer_Sim.ipynb) |
| `28` | Pipeline 的 micro-batch 时序，适合看 Step 1-4 的气泡和排布 | [28 Pipeline Parallelism MicroBatch](../02_PyTorch_Algorithms/28_Pipeline_Parallelism_MicroBatch.ipynb) |
| `29` | Tensor Parallelism 的通信开销，适合看 Step 1-4 的切分与代价 | [29 Tensor Parallelism Sim](../02_PyTorch_Algorithms/29_Tensor_Parallelism_Sim.ipynb) |
| `46` | NCCL 通信热点与等待时间，适合先看 Step 1-4 的观测流程 | [46 Communication Profiling with NCCL](../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.ipynb) |
| `79` | 分布式并行基准项目，适合先看 Step 1-4 的实验设置和结果汇总 | [79 Distributed Parallel Benchmark](../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) |

## 推荐入口

- 先看 `Part 1C`，把“为什么多卡一定会带来通信问题”先立住。
- 再看 `2.8`，把 ZeRO、Pipeline 和 Tensor Parallelism 的策略边界串起来。
- 如果想看通信代价如何被量化，再回到 `46` 和 `79`。

## 为什么这个专题不能退化成索引

如果这里只是把 `05 / 06 / 20 / 27 / 28 / 29 / 46 / 79` 列出来，它就仍然只是“去哪里看”的答案。横向专题真正应该补的是三层厚度：

- 文字串联：为什么系统会从单卡约束走向多卡切分，通信为什么会成为新的主矛盾。
- 文献锚点：这些切分方法分别是谁提出的、主要在解决哪一类瓶颈。
- 可视化：让读者看到图就知道当前是在切状态、切层、切算子，还是在对付路由热点。

所以，后面的编号页不是文件目录，而是承接主线后的故事重组。

## 入口摘要

- 第一入口：`Part 1C` + `05 -> 20 -> 06 -> 13`，先把通信原语、显存分摊和观测基础立住。
- 第二入口：`2.8 -> 27 -> 28 -> 29`，把 ZeRO、Pipeline 和 Tensor Parallelism 的主线补齐。
- 验证入口：`46 -> 79 -> 66`，把通信热点、分布式基准和最终收益连起来。

## 01-06 骨架

这 6 个小节是知识组织层，不要求和已有 notebook 一一对应。它们围绕一条主故事线展开：

- 先解释系统为什么会走向并行；
- 再解释同步和数据并行的第一层代价；
- 再解释状态切分、层切分、算子切分和 expert parallel；
- 最后回到 benchmark 和并行选型。

## 正文页

- [01 Why Parallel and Communication](./01_why_parallel_and_communication.md)
- [02 Data Parallel and Synchronization](./02_data_parallel_and_synchronization.md)
- [03 State Sharding and ZeRO](./03_state_sharding_and_zero.md)
- [04 Pipeline and Tensor Parallel](./04_pipeline_and_tensor_parallel.md)
- [05 Expert Parallel and Communication Hotspots](./05_expert_parallel_and_communication_hotspots.md)
- [06 Benchmark and Parallel Decision](./06_benchmark_and_parallel_decision.md)
- [07 Visual Assets](./07_visual_assets.md)
- [通信与并行正文](./casebook.md)：按“通信原语 / 并行切分 / 调度代价 / 基准验证”展开专题正文，适合做更细的选型和对照。
- [通信与并行深入阅读](./walkthrough.md)：按完整并行选型过程展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先确认瓶颈在通信、算子还是调度时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当并行策略和显存分摊、cache 压力一起出现时先看这里。
- [编译与图优化专题](../compiler_graph_optimization/intro.md)：当通信策略和执行模型、backend 约束一起分析时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先从 `Part 1C` 进入，把通信拓扑、显存共享和多卡边界先立住。
- 再看 `05 -> 20 -> 06 -> 13`，把通信原语、显存收益和 profiling 观测串起来。

### Part 2 入口

- 先看 `2.8 -> 27 -> 28 -> 29`，把 ZeRO、Pipeline 和 Tensor Parallelism 的主线补齐。
- 再看 `46 -> 79`，把通信 profiling 和分布式 benchmark 连接起来。
- 如果需要回看收益证明，再回到 `66` 看最终验证口径。

## 典型阅读链

- 如果你想先理解多卡通信原理，先读 `05 -> 20`，把通信拓扑和 AllReduce 先讲通。
- 如果你想先理解显存是怎么被并行策略切开的，先读 `06 -> 27`，把 ZeRO 的收益和代价讲清楚。
- 如果你想先理解流水线为什么会有气泡，先读 `28 -> 34`，把 micro-batch、排布和基准结果串起来。
- 如果你想先理解张量切分的通信压力，先读 `29 -> 46`，把切分方式和通信热点串起来。
- 如果你想先看并行策略值不值，先读 `46 -> 79 -> 66`，把通信 profile、分布式 benchmark 和最终收益连起来。

## 读法建议

- 如果你关心“通信原语怎么工作”，先看 `05 -> 20`。
- 如果你关心“多卡训练怎么切”，先看 `06 -> 27 -> 28 -> 29`。
- 如果你关心“怎么证明并行策略值不值”，先看 `46 -> 79`。
- 如果你想先补前置桥，可以先看 `Part 1C` 的 Group Overview，再回到 `05 / 06 / 20`。
- 如果你关心“如何把并行策略和性能验证连起来”，先看 `06 -> 27 -> 28 -> 29 -> 46 -> 79`。

## 建设方式

- 先补通信原语和策略边界，再补正文页里的案例、对照和误区。
- 优先从 Part 1C / Part 2.8 里抽取高频、稳定、可复用的结论。
- 让正文页专注回答“通信代价来自哪里、并行策略换来了什么”。
- 后续新增内容时，优先沿着 `通信原语 -> 并行切分 -> profiling -> benchmark` 这条线放到正文页。

## 专题状态
本专题已更新为 `01-06 + 07_visual_assets` 的解释层结构。当前已完成正文层，下一步优先补图册与更细的论文锚点。
