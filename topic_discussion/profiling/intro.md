# Profiling 专题

## 专题定位

本专题用于串起 profiling 主线：先看问题怎么被定义，再看时间热点、memory timeline、通信等待和 benchmark 验证怎样一起形成证据链，最后把结果收成 inspect / optimize / validate / revert 的行动建议。这里聚焦性能取证；如果问题已经明确变成显存预算或推理选型，应转到对应专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part00 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | profiling 与调试前置桥 | `0E -> 17` | [01 Why Profiling Matters](./01_why_profiling_matters.md) |
| Task2 | memory profiling 与异常定位 | `18 -> 19 -> 20` | [03 Memory Timeline and Residency](./03_memory_timeline_and_residency.md) |
| Task3 | 时间热点与 trace 阅读 | `20` | [02 Time Breakdown and Trace Reading](./02_time_breakdown_and_trace_reading.md) |
| Task4 | 训练 / 推理证据采集 | `19 -> 20 -> 22` | [03 Memory Timeline and Residency](./03_memory_timeline_and_residency.md) |
| Task5 | 通信与分布式 profiling | `46 -> 79` | [04 Communication Wait and Overlap](./04_communication_wait_and_overlap.md) |
| Task6 | 回归验证与行动建议 | `74` | [05 Benchmark Design and Regression Validation](./05_benchmark_design_and_regression_validation.md), [06 Diagnosis and Action Decision](./06_diagnosis_and_action_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“现在到底慢在哪里”“看到一个热点后下一步该做什么”时，再回来看对应的专题正文。想看汇总版就进 [Profiling 正文](./casebook.md)，想按连续故事线走一遍就进 [Profiling 深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[显存优化专题](../memory_performance_tuning/intro.md) 负责预算与 trade-off，[推理优化专题](../inference_optimization/intro.md) 负责请求链路判断，[通信与并行专题](../communication_parallel/intro.md) 负责多卡等待和切分代价。
