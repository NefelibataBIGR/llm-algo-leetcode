# 编译与图优化专题

## 专题定位

本专题用于串起编译与图优化主线：先看为什么“图看起来正确”不等于“跑起来高效”，再看 fusion、lowering、schedule、layout 和 backend 约束分别改的是哪一层，最后把差异收回 benchmark 和项目结论。这里聚焦图级判断和执行级约束；如果问题先表现为推理策略或多卡通信，应转到对应专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part01 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 图级判断与 fusion 直觉 | `P1:09 -> 19` | [01 Why Compiler and Graph Optimization Matters](./01_why_compiler_and_graph_optimization_matters.md) |
| Task2 | lowering、legalization 与 scheduling | `P1:08 -> 32` | [03 Lowering, Legalization and Scheduling](./03_lowering_legalization_and_scheduling.md) |
| Task3 | 执行模型与 backend 约束 | `P1:15 -> 16 -> 18` | [04 Execution Model and Backend Constraints](./04_execution_model_and_backend_constraints.md) |
| Task4 | backend 成本模型 | `P1:33` | [05 Backend Cost Models and Divergent Optima](./05_backend_cost_models_and_divergent_optima.md) |
| Task5 | 推理与图优化交叉处 | `20 -> 22 -> 34` | [02 Graph Structure and Fusion Decisions](./02_graph_structure_and_fusion_decisions.md) |
| Task6 | benchmark 与项目验证 | `66 -> 67 -> 74` | [06 Benchmark and Project Validation](./06_benchmark_and_project_validation.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“同一张图为什么在不同 backend 上差很多”“fusion 和 schedule 到底谁决定结果”时，再回来看对应的专题正文。想看汇总版就进 [编译与图优化正文](./casebook.md)，想按连续故事线走一遍就进 [编译与图优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责热点证据链，[推理优化专题](../inference_optimization/intro.md) 负责请求链路视角，[通信与并行专题](../communication_parallel/intro.md) 负责切分与通信代价。
