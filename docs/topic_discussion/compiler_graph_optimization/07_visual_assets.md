# 07 Visual Assets

## 页面目标

这页负责收口编译与图优化专题的图册资产。第一阶段先固定图的职责，后续再补正式 SVG。

## 图册顺序

1. `graph_to_backend_chain`
- 从 graph 到 lowering、execution、benchmark 的总图

![Graph to Backend Chain](/topic_discussion/compiler_graph_optimization/graph_to_backend_chain.svg)

2. `fusion_cost_map`
- 哪些中间张量和依赖让 fusion 真正有意义

![Fusion Cost Map](/topic_discussion/compiler_graph_optimization/fusion_cost_map.svg)

3. `lowering_schedule_ladder`
- legalize -> lower -> schedule -> codegen 的链路图

![Lowering and Schedule Ladder](/topic_discussion/compiler_graph_optimization/lowering_schedule_ladder.svg)

4. `backend_constraint_map`
- CUDA / Triton / layout / launch 如何限制图级选择

![Backend Constraint Map](/topic_discussion/compiler_graph_optimization/backend_constraint_map.svg)

5. `divergent_optima_board`
- 同图不同 backend 的最优解分化图

![Divergent Optima Board](/topic_discussion/compiler_graph_optimization/divergent_optima_board.svg)

6. `compiler_benchmark_decision`
- keep / tune / switch backend / reject 的决策图

![Compiler Benchmark Decision](/topic_discussion/compiler_graph_optimization/compiler_benchmark_decision.svg)

## 当前状态

第一批和第二批图已补齐，当前图册已覆盖 `01-06` 的主要入口。
