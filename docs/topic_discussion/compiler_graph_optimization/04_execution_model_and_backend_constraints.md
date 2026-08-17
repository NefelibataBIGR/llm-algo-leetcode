# 04 执行模型与 Backend 约束

## 页面目标

这一页负责解释 backend 约束如何反过来塑造图优化结果，以及为什么执行模型必须进到编译讨论里。

## 问题起点

很多图级判断在 backend 级别会遇到现实约束：

- block / warp / program shape
- layout 选择
- stream / launch 组织
- kernel 粒度和资源占用

## 关键观察点

- CUDA / Triton execution model
- layout-sensitive kernel structure
- launch granularity
- backend-specific constraints

## 可视化入口

![Backend Constraint Map](/topic_discussion/compiler_graph_optimization/backend_constraint_map.svg)

## 对应 Part

- `08 Programming Models CUDA Triton`
- `15 CUDA Execution Model`
- `18 Triton Block Model`
- `29 CUDA Stream Advanced Scheduling`

## 小结

backend 约束不是图优化之后才补看的细节，而是会提前决定哪些图级选择真正可用。
