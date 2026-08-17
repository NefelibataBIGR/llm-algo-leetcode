# 03 Memory Timeline 与 Residency

## 页面目标

这一页负责解释 memory timeline 在 profiling 里扮演什么角色，以及它和显存优化专题中的预算决策有什么区别。

## 问题起点

很多性能问题表面上是“慢”，但背后其实和内存行为有关：

- allocation 频繁震荡
- 某类对象驻留过久
- activation / cache 把阶段切换拖慢

## profiling 视角

在 profiling 里看 memory timeline，是为了回答：

- 哪个阶段内存突然抬高？
- 这次抬高是否和时间热点同步？
- residency pattern 是否说明系统在等内存行为？

这和显存优化专题里的“怎么压预算”不同。

## 可视化入口

![Memory Timeline Diagnosis](/topic_discussion/profiling/memory_timeline_diagnosis.svg)

## 对应 Part

- `18 Memory Profiling and Optimization`
- `19 Activation Checkpointing and Activation Offload`
- `20 Profiling and Memory Ledger`

## 小结

profiling 里的 memory timeline 首先是证据，不是优化动作本身。
