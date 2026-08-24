# 性能分析（Profiling）专题

> 导读：这个专题不从工具按钮出发，而是训练你把“哪里慢、为什么慢、改完是否真的更好”串成一条可复现的证据链，再决定回到推理、显存、通信或算子层继续优化。

## 专题定位与 Infra 层定位

本专题串起 profiling 主线：先定义问题，再用时间热点、memory timeline、通信等待和 benchmark 验证形成证据链，最后收成 `inspect / optimize / validate / revert` 的行动建议。Profiling 贯穿 Infra-L1–Infra-L5，是证据方法而不是独立软件层：Infra-L1 看硬件利用率与带宽，Infra-L2 看 kernel、算子库、通信库和编译结果，Infra-L3 看框架与运行时，Infra-L4 看服务请求、KV Cache 和吞吐，Infra-L5 看资源调度与回归治理。

它的作用是把计算、内存、通信三条能力轴放到同一条时间线上，再决定应该回到哪一层优化。若问题已经明确变成显存预算或推理选型，应转到对应专题；若已定位到算子或图融合，则转到编译与图优化专题。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的性能与项目路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 74、79 等项目节。73、76、75 属于显存优化路线的训练侧项目，Profiling 在其中提供证据方法；74 是显存路线的最终收口项目，同时复用本专题的方法。

## 前置阅读

建议先掌握 `Part 00: 0E` 的调试基础、`Part 01: 13` 的系统与性能认知，再进入 Part 02 的训练、推理或并行项目。若只想定位单个性能问题，可直接从下面的 Task 对应正文开始。

## 工具分层与环境边界

本专题不要求一开始安装完整的 GPU 工具链。工具应随着问题粒度逐级增加：先用轻量测量确认现象，再用框架级 profiler 找方向，只有在需要解释系统重叠或 kernel 细节时，才进入 Nsight 和分布式工具。

| 层级 | 工具 | 主要回答的问题 | 环境要求 |
|:---|:---|:---|:---|
| Level 0：轻量测量 | `time.perf_counter()`、`torch.cuda.Event`、`torch.cuda.synchronize()`、`nvidia-smi` | 总耗时、GPU 计时、峰值显存和进程状态 | CPU 可做部分验证；GPU 计时需要 CUDA |
| Level 1：框架级 | `torch.profiler`、Chrome Trace、TensorBoard | 时间热点、CPU/GPU 时间线、算子排序和训练阶段 | CPU 可运行基础示例；CUDA trace 需要 GPU |
| Level 2：系统级 | Nsight Systems | CPU-GPU overlap、stream、同步点、数据搬运和服务阶段 | NVIDIA GPU、Nsight Systems |
| Level 3：kernel / 分布式级 | Nsight Compute、NCCL trace / debug log | occupancy、访存吞吐、Tensor Core、通信等待和 overlap | GPU；多卡分析还需要分布式环境 |

`73`、`76`、`75` 是显存优化路线的训练侧项目，分别负责基线、策略比较和预算决策；`74` 使用 Level 0-2 的证据对显存优化方案做最终端到端收口。`79` 和 `46` 延伸到 Level 3 的通信与并行问题。Colab / ModelScope 学习者完成 Level 0-1 即可，Level 2-3 标记为 GPU 服务器扩展，不作为主线前置。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 00 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | profiling 与调试前置桥 | `0E -> 17` | [01 Why Profiling Matters](./01_why_profiling_matters.md) |
| Task2 | 时间热点与 trace 阅读 | `20` | [02 Time Breakdown and Trace Reading](./02_time_breakdown_and_trace_reading.md) |
| Task3 | memory profiling 与异常定位 | `18 -> 19 -> 20` | [03 Memory Timeline and Residency](./03_memory_timeline_and_residency.md) |
| Task4 | 训练 / 推理证据采集与通信等待 | `19 -> 20 -> 22 -> 46` | [04 Communication Wait and Overlap](./04_communication_wait_and_overlap.md) |
| Task5 | benchmark 设计与回归验证 | `46 -> 79 -> 74` | [05 Benchmark Design and Regression Validation](./05_benchmark_design_and_regression_validation.md) |
| Task6 | 回归验证与行动建议 | `74` | [06 Diagnosis and Action Decision](./06_diagnosis_and_action_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“现在到底慢在哪里”“看到一个热点后下一步该做什么”时，再回来看对应的专题正文。想看汇总版就进 [Profiling 正文](./casebook.md)，想按连续故事线走一遍就进 [Profiling 深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[显存优化专题](../memory_performance_tuning/intro.md) 负责预算与 trade-off，[推理优化专题](../inference_optimization/intro.md) 负责请求链路判断，[通信与并行专题](../communication_parallel/intro.md) 负责多卡等待和切分代价。

## 项目结论

Profiling 复用的显存项目闭环是 `73 训练性能基线 -> 76 策略比较 -> 75 预算决策 -> 74 最终验证`；独立的通信与并行扩展再进入 `79`。项目结果应至少保留 workload、环境、基线、优化策略、指标和结论，避免只凭单次运行日志下判断。

## 环境与验证

基础 trace 阅读和部分模拟实验可先用 CPU；真实 GPU profiling、显存时间线和多卡通信需要对应 GPU 或分布式环境。建议固定 workload、warmup、迭代次数和随机种子，并将结果保存为 JSON；跨机器比较时同时记录 PyTorch、CUDA、驱动、GPU 型号和并行配置。
