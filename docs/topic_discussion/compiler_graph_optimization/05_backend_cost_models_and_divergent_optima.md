# 05 Backend 成本模型与最优解分化

## 页面目标

这一页负责解释为什么同一张图在不同 backend 上会产生不同最优解，以及成本模型为什么必须进入判断。

## 问题起点

如果只看语义，很多方案都“正确”。  
真正让方案分化的是：

- 目标硬件不同
- 带宽、寄存器、缓存和 launch 成本不同
- 编译器和 runtime 的假设不同

## 关键观察点

- backend cost model
- hardware-specific assumptions
- TCO and deployment constraints
- divergent optima

## 可视化入口

![Divergent Optima Board](/topic_discussion/compiler_graph_optimization/divergent_optima_board.svg)

## 对应 Part

- `33 TCO and Cost Model`
- `09 AI Compilers and Graph Optimization`
- `32 TVM MLIR Deep Practice`

## 小结

backend 差异不是噪声，而是“为什么最优解会分化”的主因。
