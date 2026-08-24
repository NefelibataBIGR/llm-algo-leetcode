# Profiling 专题

## 专题定位

本专题用于串起 profiling 主线：先看问题怎么被定义，再看时间热点、memory timeline、通信等待和 benchmark 验证怎样一起形成证据链，最后把结果收成 inspect / optimize / validate / revert 的行动建议。这里聚焦性能取证；如果问题已经明确变成显存预算或推理选型，应转到对应专题。

## Infra 层定位

Profiling 是贯穿 L1-L5 的证据层，而不是某一个独立软件层：L1 看硬件利用率与带宽，L2 看 kernel、算子库、通信库和编译结果，L3 看框架与运行时，L4 看服务请求、KV cache 和吞吐，L5 看资源调度与回归评测。它的作用是把计算、内存、通信三条能力轴放到同一条时间线上，再决定应该回到哪一层优化。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的性能与项目路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 73、74、76、79 等项目节。Profiling 是横向支撑专题，不要求从专题正文第一页顺序开始。

## 前置阅读

建议先掌握 `Part 00: 0E` 的调试基础、`Part 01: 13` 的系统与性能认知，再进入 Part 02 的训练、推理或并行项目。若只想定位单个性能问题，可直接从下面的 Task 对应正文开始。

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

推荐的实践闭环是 `73 训练性能分析 -> 74 Profiling 驱动的端到端优化 -> 79 分布式并行 benchmark`；如果问题集中在激活、检查点或 offload，则接入 `76`。项目结果应至少保留 workload、环境、基线、优化策略、指标和结论，避免只凭单次运行日志下判断。

## 环境与验证

基础 trace 阅读和部分模拟实验可先用 CPU；真实 GPU profiling、显存时间线和多卡通信需要对应 GPU 或分布式环境。建议固定 workload、warmup、迭代次数和随机种子，并将结果保存为 JSON；跨机器比较时同时记录 PyTorch、CUDA、驱动、GPU 型号和并行配置。
