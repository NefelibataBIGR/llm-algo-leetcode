# 06 Benchmark 与项目验证

## 页面目标

这一页负责把编译和 backend 结论收束到 benchmark 与项目验证：到底哪种执行方案值得保留。

## 决策框架

1. 先确认 graph / lowering / backend 假设是否一致。
2. 再确认 benchmark 是否覆盖真实 workload。
3. 最后判断是：
   - `keep`
   - `tune`
   - `switch backend`
   - `reject`

## 可视化入口

![Compiler Benchmark Decision](/topic_discussion/compiler_graph_optimization/compiler_benchmark_decision.svg)

## 对应 Part

- `2.2`
- `2.6`
- `2.7`
- `2.10`

## 小结

编译专题的终点不是“生成了更复杂的执行链”，而是“项目结果是否真的更好”。
