# 02 时间拆分与 Trace 阅读

## 页面目标

这一页负责把“为什么慢”先拆成可观察的时间问题：operator、kernel、launch、等待和阶段切换。

## 问题起点

“慢”不是一个统一现象。常见情况包括：

- 单个 operator 真慢
- kernel 数量太碎
- Python / runtime 开销高
- 某些阶段有明显等待

如果不先做时间拆分，后面的优化动作常常会打偏。

## 关键观察点

- operator breakdown
- kernel timeline
- launch overhead
- step 内阶段切换
- CPU / GPU overlap

## 可视化入口

![Time Breakdown and Trace Reading](/topic_discussion/profiling/time_breakdown_trace.svg)

## 对应 Part

- `17 PyTorch Profiling Basics`
- `20 Profiling and Memory Ledger`
- `74 Profiling Driven End-to-End Optimization`

## 小结

时间拆分是 profiling 的第一入口，因为大多数问题都先表现为“某一段时间不合理”。
