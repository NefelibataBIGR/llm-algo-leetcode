# 06 从诊断到行动决策

## 页面目标

这一页负责把 profiling 的证据链收束成行动：继续观察、深入排查、优化、还是回退。

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

## 小结

profiling 的终点不是图，而是可执行行动。
