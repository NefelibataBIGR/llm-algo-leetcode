# 06 从诊断到行动决策

## 页面目标

这一页负责把 profiling 的证据链收束成行动：继续观察、深入排查、优化、还是回退。

本页的输出不是“最优方案”，而是带证据等级的下一步动作。只有问题定义、profile 证据和 benchmark 结果相互支持时，才应把优化提升为项目结论。

## 决策框架

1. 先确认问题类别：compute、memory、communication、experiment design。
2. 再确认证据是否足够稳定。
3. 再决定动作：
   - `keep observing`
   - `inspect deeper`
   - `optimize`
   - `revert`

## 可视化入口

![Diagnosis and Action Decision](/topic_discussion/profiling/action_decision_board.svg)

## 常见误区

- 有热点就改
- 只看一次 trace
- 没有 before / after 对照就下结论

## 项目结论

profiling 的终点不是图，而是可执行行动。

## 回到项目

需要训练侧性能实验时进入 `73 -> 74`；需要显存策略比较时进入 `76 -> 75`；需要多卡验证时进入 `79 -> 80 -> 81`。这些项目都应复用本专题的证据链，而不是只复制某个 profiler 命令。
