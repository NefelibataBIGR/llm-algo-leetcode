# Profiling 正文

这页只做 profiling 问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 判断表

先分清问题在时间热点、memory timeline、通信等待还是 benchmark 验证，再判断证据链是否足够支撑下一步动作。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 系统变慢了，但不知道慢在哪 | `time hotspot` | [02](./02_time_breakdown_and_trace_reading.md) | 先看 operator、trace、阶段拆分 |
| 时间和显存一起波动 | `memory residency` | [03](./03_memory_timeline_and_residency.md) | 看 allocation、residency、timeline |
| 多卡收益不稳 | `wait / overlap mismatch` | [04](./04_communication_wait_and_overlap.md) | 看同步等待、communication trace |
| before / after 结果说不清 | `benchmark gap` | [05](./05_benchmark_design_and_regression_validation.md), [06](./06_diagnosis_and_action_decision.md) | 回到 workload 和回归验证 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 时间热点 | 慢点是不是已经被定位清楚 | 看一眼 trace 就下结论 |
| memory timeline | 时间问题是不是伴随显存驻留问题 | 看显存高就直接去省显存 |
| communication wait | 多卡是不是在等而不是在算 | 多卡慢就一定是算子问题 |
| benchmark | 这次优化是不是可复现、可比较 | before / after 口径不统一 |

## 小结

这页的职责不是教工具按钮，而是把 profiling 里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。
