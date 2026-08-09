# 03 Lowering、Legalization 与 Scheduling

## 页面目标

这一页负责解释为什么 lowering 不是简单翻译，以及 legal / executable / efficient 这三件事为什么不能混成一个判断。

## 问题起点

一个高层图即使语义上合理，也还要回答：

- 能不能合法地下沉到目标表示？
- 下沉之后是否仍然适合调度？
- schedule 是否把图级收益保住了？

## 关键观察点

- legalization
- schedule search / manual schedule
- layout transformation
- codegen 之前的执行形态

## 可视化入口

![Lowering and Schedule Ladder](/topic_discussion/compiler_graph_optimization/lowering_schedule_ladder.svg)

## 对应 Part

- `32 TVM MLIR Deep Practice`
- `33 TCO and Cost Model`
- `29 CUDA Stream Advanced Scheduling`

## 小结

lowering 的关键不是“能翻译”，而是“翻译之后是否仍然划算”。
