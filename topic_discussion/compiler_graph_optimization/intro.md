# 编译与图优化专题

## 专题定位与 Infra 层定位

本专题是“算子与编译优化”路线的入口，内部包含 kernel / 算子优化和编译 / 图优化两条分支（建设中）。学习顺序是：先看为什么“图看起来正确”不等于“跑起来高效”，再看 fusion、lowering、schedule、layout 和 backend 约束分别改的是哪一层，最后把差异收回 benchmark 和项目结论。

本专题主要连接 Infra-L2–Infra-L3，并向下受 Infra-L1 硬件约束、向上服务于 Infra-L4 的训练和推理运行时：算子库、编译器、kernel 选择和 backend lowering 属于 Infra-L2，框架图、执行计划与运行时调度属于 Infra-L3。它不替代推理、显存或通信专题，而是解释同一策略如何经过编译后改变计算、内存访问和通信成本。

## 推荐入口

推荐从 [推理优化专题](../inference_optimization/intro.md) 的请求链路和算子基础进入，再在遇到 kernel、fusion 或 backend 差异时回看本专题。需要证据采集时，与 [Profiling 专题](../profiling/intro.md) 配合使用。

## 前置阅读

建议先掌握 `Part 01` 的 GPU 执行与算子基础，再阅读表中的 lowering、schedule 和 backend 相关来源。初学者可先看 Task1-2 建立图到 kernel 的映射，再进入项目 benchmark。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 图级判断与 fusion 直觉 | `Part 01:09 -> 19` | [01 Why Compiler and Graph Optimization Matters](./01_why_compiler_and_graph_optimization_matters.md) |
| Task2 | lowering、legalization 与 scheduling | `Part 01:08 -> 32` | [03 Lowering, Legalization and Scheduling](./03_lowering_legalization_and_scheduling.md) |
| Task3 | 执行模型与 backend 约束 | `Part 01:15 -> 16 -> 18` | [04 Execution Model and Backend Constraints](./04_execution_model_and_backend_constraints.md) |
| Task4 | backend 成本模型 | `Part 01:33` | [05 Backend Cost Models and Divergent Optima](./05_backend_cost_models_and_divergent_optima.md) |
| Task5 | 推理与图优化交叉处 | `20 -> 22 -> 34` | [02 Graph Structure and Fusion Decisions](./02_graph_structure_and_fusion_decisions.md) |
| Task6 | benchmark 与项目验证 | `66 -> 67 -> 74` | [06 Benchmark and Project Validation](./06_benchmark_and_project_validation.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“同一张图为什么在不同 backend 上差很多”“fusion 和 schedule 到底谁决定结果”时，再回来看对应的专题正文。想看汇总版就进 [编译与图优化正文](./casebook.md)，想按连续故事线走一遍就进 [编译与图优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责热点证据链，[推理优化专题](../inference_optimization/intro.md) 负责请求链路视角，[通信与并行专题](../communication_parallel/intro.md) 负责切分与通信代价。

## 项目结论

推荐以 `66 推理性能比较 -> 67 量化推理与部署 -> 74 Profiling 驱动优化` 形成最小验证闭环。结论至少应同时记录图或 kernel 的变化、端到端延迟、吞吐、显存和 backend 环境；单个算子变快不等于服务整体变快。

## 环境与验证

图结构、成本模型和部分 lowering 模拟可用 CPU；真实编译、kernel autotune 和 serving backend 验证通常需要 GPU。应固定输入形状、warmup、迭代次数和后端版本，并保留编译日志与 benchmark JSON。
