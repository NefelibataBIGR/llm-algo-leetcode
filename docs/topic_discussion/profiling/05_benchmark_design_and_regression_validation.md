# 05 Benchmark 设计与回归验证

## 页面目标

这一页负责解释：就算定位出了热点，也不能直接宣布优化成立，必须回到 benchmark 和回归验证。

本页的输出是可比较的 before / after 结果：固定 workload、环境、warmup 和统计口径，并同时记录收益、代价与波动。没有这一步，profiling 只能提供线索，不能提供项目结论。

## 问题起点

profiling 最容易出现的假结论有两类：

- 单次 trace 看起来好了，但 workload 不一致
- 某个指标变好了，但整体系统没变好

## 关键问题

- before / after 是否可比
- 请求分布或 batch 口径是否一致
- 波动是否可接受
- 回归是否可重复

## 可视化入口

![Benchmark Validation Board](/topic_discussion/profiling/benchmark_validation_board.svg)

## 对应 Part

- `74 Profiling Driven End-to-End Optimization`
- `79 Distributed Parallel Benchmark`
- `66 Inference Performance Comparison`

## 本节要点

profiling 的结论必须回到 benchmark 才算完成，否则它只是一次局部观察。

## 进入下一页

将 benchmark 结果交给 [06 从诊断到行动决策](./06_diagnosis_and_action_decision.md)，判断是保留优化、继续采证、扩大实验还是回退。
