# Profiling 专题

## 专题概览

本专题用于把 `Part 00-02` 里分散的 profiling、benchmark、trace、memory timeline 和回归验证内容重组为一条**性能取证故事线**，回答三个核心问题：

- 现在到底慢在哪里，证据是什么？
- 看到一个热点之后，怎样区分它是算子问题、访存问题、同步等待，还是实验设计问题？
- profiling 结果如何落成可执行结论，而不是“看过一堆图”？

这条线承接 `0E`、`Part 1` 和 `Part 2` 的已有学习路线，但不复述目录。横向专题负责把这些素材重组成“问题提出 -> 采集证据 -> 归因 -> 验证 -> 决策”的知识骨架。

## 职责边界

`profiling` 和 `显存优化与性能调优` 都会看性能图、显存图和 benchmark，但两者的主问题不同：

- `profiling` 先问：**现在慢在哪里，证据链是否成立？**
- `显存优化与性能调优` 先问：**为什么装不下，怎样把峰值显存压进预算？**

更具体地说：

- `01` 负责建立 profiling 的目标、口径和误区。
- `02` 负责时间热点、算子拆分和 trace 阅读。
- `03` 负责 memory timeline、allocation pattern 和资源取证。
- `04` 负责通信等待、overlap 和分布式 trace。
- `05` 负责 benchmark 设计、回归验证和实验口径。
- `06` 负责从证据走向行动，输出 keep / inspect / optimize / revert。
- `07` 负责图册收口。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 0E` | profiling 与调试的前置桥 |
| `Part 1A-1E` | 资源账本、访存、通信、执行模型和 backend 背景 |
| `Part 2.5 / 2.6 / 2.8` | 训练、推理和分布式里的真实性能现象 |
| `17 / 18 / 20 / 46 / 74 / 79` | profiling、memory、benchmark 和通信热点的主要样本 |

## Task1-6 路线

`Task1-6` 继续保留为学习内容路径；`01-06` 是知识组织层。二者并存，不要求一一对应。

| Task | 学习内容 | 章节 |
|:---|:---|:---|
| Task1 | profiling 与调试前置桥 | [0E](../../00_Prerequisites/0E.md)、[17 PyTorch Profiling Basics](../../00_Prerequisites/17_PyTorch_Profiling_Basics.md) |
| Task2 | memory profiling 与异常定位 | [18 Memory Profiling and Optimization](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)、[19 Debugging and Anomaly Localization](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.md) |
| Task3 | profiling 与 memory ledger 入口 | [20 Profiling and Memory Ledger](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.md) |
| Task4 | 训练 / 推理场景中的证据采集 | [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md)、[20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)、[22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md) |
| Task5 | 通信与分布式 profiling | [46 Communication Profiling with NCCL](../../02_PyTorch_Algorithms/46_Communication_Profiling_with_NCCL.md)、[79 Distributed Parallel Benchmark](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.md) |
| Task6 | 端到端验证与决策 | [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)、[06 Diagnosis and Action Decision](./06_diagnosis_and_action_decision.md) |

## 01-06 骨架

这 6 个编号页是专题正文，不是文件索引。它们围绕“性能问题如何被证明，而不是被猜测”来组织。

| 章节 | 你会得到什么 | 适合先从哪里进入 |
|:---|:---|:---|
| `01` | profiling 目标、证据链和常见误判 | 先想弄清楚 profiling 到底是干什么 |
| `02` | 时间热点、operator table、trace 读取 | 先看为什么慢 |
| `03` | memory timeline、allocation 和 residency | 先看是不是 memory 路线导致问题 |
| `04` | communication wait、overlap 和多卡热点 | 先看是不是同步与通信问题 |
| `05` | benchmark 设计、回归验证和实验口径 | 先看如何证明优化真的成立 |
| `06` | 从 profiling 证据到行动决策 | 已经有 trace，想知道接下来该改什么 |

## 文献锚点

- PyTorch Profiler 官方文档与 trace 解释资料。
- CUDA / kernel timeline / operator breakdown 相关资料。
- NCCL 和分布式 trace 相关资料。
- benchmark 设计与性能回归验证资料。

## 推荐入口

- 如果你第一次接触 profiling，先看 `01 -> 02`。
- 如果你已经有 trace，但不知道怎么归因，先看 `03 -> 04`。
- 如果你最关心“优化到底算不算成立”，先看 `05 -> 06`。

## 正文页

- [01 Why Profiling Matters](./01_why_profiling_matters.md)
- [02 Time Breakdown and Trace Reading](./02_time_breakdown_and_trace_reading.md)
- [03 Memory Timeline and Residency](./03_memory_timeline_and_residency.md)
- [04 Communication Wait and Overlap](./04_communication_wait_and_overlap.md)
- [05 Benchmark Design and Regression Validation](./05_benchmark_design_and_regression_validation.md)
- [06 Diagnosis and Action Decision](./06_diagnosis_and_action_decision.md)
- [07 Visual Assets](./07_visual_assets.md)
- [Profiling 正文](./casebook.md)
- [Profiling 深入阅读](./walkthrough.md)

## 相关专题

- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当问题重点变成资源预算和 trade-off 时看这里。
- [推理优化专题](../inference_optimization/intro.md)：当 profiling 指向 prefill、decode、cache 或调度时看这里。
- [通信并行专题](../communication_parallel/intro.md)：当 profiling 指向同步等待、overlap 和多卡切分时看这里。

## 专题状态

当前专题已开始按 `01-06 + 07_visual_assets` 重构。下一步优先补编号正文页、再补第一批 SVG 图。
