# 02 图结构与 Fusion 决策

## 页面目标

这一页负责把 graph optimization 的第一层拆清楚：哪些依赖、节点和中间张量真的值得改写或 fuse。

## 问题起点

图优化最容易被简化成“能 fuse 就 fuse”。  
但真正的问题是：

- 哪些中间结果真的贵
- 哪些依赖会让某种融合失去意义
- 哪些图级重排只是把成本换了个地方

## 关键观察点

- cost vector
- dependency structure
- intermediate tensors
- layout-sensitive fusion

## 可视化入口

![Fusion Cost Map](/topic_discussion/compiler_graph_optimization/fusion_cost_map.svg)

## 对应 Part

- `09 AI Compilers and Graph Optimization`
- `19 Operator Fusion Introduction`
- `33 TCO and Cost Model`

## 小结

图级判断的重点不是“变少”，而是“变少之后执行成本是否真的下降”。
