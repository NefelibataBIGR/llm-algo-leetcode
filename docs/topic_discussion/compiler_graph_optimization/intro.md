# 编译与图优化专题

## 专题概览

本专题用于把 `Part 01-02` 里分散的图优化、融合、lowering、调度和 backend 约束重组为一条**从高层图到高效执行**的故事线，回答三个核心问题：

- 为什么“图看起来没问题”并不等于“跑起来就高效”？
- 图优化、fusion、lowering、schedule、codegen 分别解决哪一段问题？
- 为什么同一张图在不同 backend 上会得到不同结果，最后又该如何回到 benchmark 和项目结论？

这条线承接 `Part 1D / 1E` 和 `Part 2` 的已有学习路线，但不复述目录。横向专题负责把这些素材重组成“图级判断 -> lowering -> 执行模型 -> backend 约束 -> benchmark 收口”的知识骨架。

## 职责边界

这个专题只负责**编译与图优化视角**，不负责推理策略本身，也不替代 profiling 或多卡并行专题。

- `01` 解释为什么图优化值得单独成章，以及图级和执行级为什么不能混为一谈。
- `02` 解释图结构、依赖、cost vector 与 fusion 判断。
- `03` 解释 lowering、legalization 和 schedule 为什么不是“翻译”。
- `04` 解释执行模型、layout、kernel 组织和 backend 约束怎样反过来塑造图优化结果。
- `05` 解释 backend 差异和成本模型为什么会让同一张图得到不同最优解。
- `06` 负责把编译判断收束到 benchmark 和项目决策。
- `07` 负责图册收口。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1E` | 图优化、编译器、TVM / MLIR、成本模型 |
| `Part 1D` | CUDA / Triton 编程模型、执行模型、stream 调度 |
| `Part 2.2` | 模型结构与实现形式如何影响执行路径 |
| `Part 2.6 / 2.7A` | 推理侧 backend、cache、调度与图优化相遇的地方 |
| `Part 2.9` | benchmark / 项目结果如何验证 backend 结论 |

## Task1-6 路线

`Task1-6` 继续保留为学习内容路径；`01-06` 是知识组织层。二者并存，不要求一一对应。

| Task | 学习内容 | 章节 |
|:---|:---|:---|
| Task1 | 图优化与 AI 编译器入口 | [09 AI Compilers and Graph Optimization](../../01_Hardware_Math_and_Systems/09_AI_Compilers_and_Graph_Optimization.md) |
| Task2 | 融合与中间张量成本 | [19 Operator Fusion Introduction](../../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md) |
| Task3 | TVM / MLIR lowering 与 codegen | [32 TVM MLIR Deep Practice](../../01_Hardware_Math_and_Systems/32_TVM_MLIR_Deep_Practice.md) |
| Task4 | 成本模型与芯片 / backend 选择 | [33 TCO and Cost Model](../../01_Hardware_Math_and_Systems/33_TCO_and_Cost_Model.md) |
| Task5 | CUDA / Triton 执行模型与调度 | [08 Programming Models CUDA Triton](../../01_Hardware_Math_and_Systems/08_Programming_Models_CUDA_Triton.md)、[15 CUDA Execution Model](../../01_Hardware_Math_and_Systems/15_CUDA_Execution_Model.md)、[18 Triton Block Model](../../01_Hardware_Math_and_Systems/18_Triton_Block_Model.md)、[29 CUDA Stream Advanced Scheduling](../../01_Hardware_Math_and_Systems/29_CUDA_Stream_Advanced_Scheduling.md) |
| Task6 | 推理 / 项目中的 backend 验证 | [2.6](../../02_PyTorch_Algorithms/2_6.md)、[2.7A](../../02_PyTorch_Algorithms/2_7A.md)、[2.9](../../02_PyTorch_Algorithms/2_9.md) |

## 01-06 骨架

这 6 个编号页是专题正文，不是文件索引。它们围绕“高层图为什么无法自动变成高效执行”来组织。

| 章节 | 你会得到什么 | 适合先从哪里进入 |
|:---|:---|:---|
| `01` | 图优化为什么值得单独看 | 先想弄清楚编译专题在解决什么 |
| `02` | 图结构、融合和依赖成本 | 先看哪些节点真的值得 fuse 或改写 |
| `03` | lowering、legalization、schedule | 先看为什么 codegen 不是最终答案 |
| `04` | 执行模型与 backend 约束 | 先看为什么同一个 graph 在不同执行层表现不同 |
| `05` | backend 差异与成本模型 | 先看为什么不同平台会有不同最优解 |
| `06` | benchmark 与项目收口 | 已经有 backend 假设，想知道怎样验证它 |

## 文献锚点

- AI compiler / graph optimization 综述与系统论文。
- TVM / MLIR / lowering 相关资料。
- operator fusion / layout / schedule 相关论文或官方文档。
- backend cost model / TCO 相关资料。

## 推荐入口

- 如果你第一次接触这条线，先看 `01 -> 02`。
- 如果你已经知道图优化概念，但搞不清 lowering / schedule，先看 `03 -> 04`。
- 如果你最关心“为什么不同 backend 结果不一样”，直接看 `05 -> 06`。

## 正文页

- [01 Why Compiler and Graph Optimization Matters](./01_why_compiler_and_graph_optimization_matters.md)
- [02 Graph Structure and Fusion Decisions](./02_graph_structure_and_fusion_decisions.md)
- [03 Lowering Legalization and Scheduling](./03_lowering_legalization_and_scheduling.md)
- [04 Execution Model and Backend Constraints](./04_execution_model_and_backend_constraints.md)
- [05 Backend Cost Models and Divergent Optima](./05_backend_cost_models_and_divergent_optima.md)
- [06 Benchmark and Project Validation](./06_benchmark_and_project_validation.md)
- [07 Visual Assets](./07_visual_assets.md)
- [编译与图优化正文](./casebook.md)
- [编译与图优化深入阅读](./walkthrough.md)

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先证明图级或 backend 级问题是否真的存在时看这里。
- [推理优化专题](../inference_optimization/intro.md)：当 backend 选择直接影响 prefill、decode 和 cache 路径时看这里。
- [通信并行专题](../communication_parallel/intro.md)：当执行模型和并行切分一起决定结果时看这里。

## 专题状态

当前专题已开始按 `01-06 + 07_visual_assets` 重构。下一步优先补编号正文页，再补第一批 SVG 图。
