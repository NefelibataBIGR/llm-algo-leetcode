<h1 align="center">llm-algo-leetcode | 大模型算法与系统教程</h1>
<p align="center">Notebook-first tutorial for LLM algorithms and systems.<br>面向大模型算法与系统的 Notebook-first 教程。</p>
<p align="center">
  学习路线：
  <a href="./topic_discussion/fine_tuning_training/intro.md">训练微调</a> /
  <a href="./topic_discussion/inference_optimization/intro.md">推理优化</a> /
  <a href="./topic_discussion/memory_performance_tuning/intro.md">显存优化</a>
  <br>
  横向专题：
  <a href="./topic_discussion/quantization/intro.md">量化与压缩</a> /
  <a href="./topic_discussion/profiling/intro.md">Profiling</a> /
  <a href="./topic_discussion/communication_parallel/intro.md">通信与并行</a>
</p>


[中文版 (Chinese)](#中文版) | [English Version](#english-version)

---

# 中文版

## 🎯 项目简介

这是一个面向大模型入门到进阶的算法实战教程，以 LLM 为主线，帮助读者通过可运行、可验证、可回顾的 Notebook，从“会看”走到“会写、会调、会优化”。A practical tutorial with theory, walkthroughs, test cases, and solutions.

### ✨ 项目特点

1. **主线清晰**：从基础能力到 Triton / CUDA 系统优化，形成完整学习链。
2. **工程导向**：以 Notebook 实战为载体，强调动手实现与性能意识。
3. **覆盖广泛**：从 PyTorch、Transformer 到推理优化、显存管理与底层实现都有对应内容。

### 👥 适合对象

- **求职面试者**：巩固 LLM 算法工程师、AI 架构师、算子开发工程师的高频考点。
- **AI 研发人员**：从代码底层理解显存优化、分布式通信与 Triton/CUDA 算子。

## 🧭 专题快捷入口

如果你不想从 Part 顺序硬读，可以先按学习路线或横向专题进入。

学习路线：

- [训练微调路线](./topic_discussion/fine_tuning_training/intro.md)：`09-13`、`30-32`、`60-65`
- [推理优化路线](./topic_discussion/inference_optimization/intro.md)：`20-25`、`34-39`、`66-70`
- [显存优化路线](./topic_discussion/memory_performance_tuning/intro.md)：`19`、`25`、`40-45`、`73-76`

横向专题：

- [量化与压缩专题](./topic_discussion/quantization/intro.md)
- [通信与并行专题](./topic_discussion/communication_parallel/intro.md)
- [Profiling 专题](./topic_discussion/profiling/intro.md)
- [后训练与对齐专题](./topic_discussion/post_training_alignment/intro.md)

## 🌐 教程总览

这套教程分为纵深主线、学习路线、横向专题和共学沉淀四层：`Part 0` 和 `Part 1` 是共同前置，`Part 2 -> Part 5` 是主线实战层，`topic_discussion` 负责把训练、推理、显存、量化、并行、profiling 这类跨 Part 主题重新串成可跳读的路线，`team_study` 则单独作为动态共学沉淀层，当前主要对应 Part 2。整体关系可以理解为前置打底 -> PyTorch 主线 -> Triton -> CUDA，学习路线负责给 Part02 收口，横向专题负责补方法论和案例闭环。

纵向主线负责把知识按层次搭起来，保证学习路径完整、能力递进清晰；学习路线负责把 Part02 的核心任务带和项目带直接连起来；横向专题负责把分散在不同 Part 里的方法论和案例重新串联起来，补足故事性、整体性和跨章节的理解闭环。

![教程总览保底图](./docs/image-1.png)


### 📚 资产总览

这套教程不要求从 `00` 开始按顺序硬读。`00` 主要是前置补齐区，如果你已有基础，可以直接从最相关的部分开始；下面这张表会直接告诉你：每一部分学什么、包含哪些组、适合谁、当前进度如何。

| 部分 | 组别 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| [`第零部分：前置知识与环境准备（5 组 / 20 节，已完成，持续优化）`](./00_Prerequisites/intro.md) | [`0A Python 基础与数据表示（4 节）`](./00_Prerequisites/0A.md) / [`0B PyTorch 张量与自动求导（4 节）`](./00_Prerequisites/0B.md) / [`0C PyTorch 模型构建（4 节）`](./00_Prerequisites/0C.md) / [`0D 训练与模型直觉（4 节）`](./00_Prerequisites/0D.md) / [`0E 调试与性能（4 节）`](./00_Prerequisites/0E.md) | 把 Python、NumPy、PyTorch、训练循环、调试工具和性能意识搭好。 | 第一次进入教程、需要补齐入门前置的人。 | ✅ 已完成，持续优化 |
| [`第一部分：硬件、数学与系统（5 组 / 33 节，已完成，持续优化）`](./01_Hardware_Math_and_Systems/intro.md) | [`1A 数值基础与算力估算（4 节）`](./01_Hardware_Math_and_Systems/1A.md) / [`1B 单卡硬件与访存优化（5 节）`](./01_Hardware_Math_and_Systems/1B.md) / [`1C 多卡通信与显存共享（5 节）`](./01_Hardware_Math_and_Systems/1C.md) / [`1D 异构调度与算子编程（5 节）`](./01_Hardware_Math_and_Systems/1D.md) / [`1E 编译优化与硬件生态（4 节）`](./01_Hardware_Math_and_Systems/1E.md) | 理解硬件、算力、访存、通信和调度这些底层约束。 | 想先弄清“为什么要这样写”和“为什么要这样部署”的学习者。 | ✅ 已完成，持续优化 |
| [`第二部分：PyTorch 算法实战（10 组，已完成，持续优化）`](./02_PyTorch_Algorithms/intro.md) | [`2.1 基础算子`](./02_PyTorch_Algorithms/intro.md) / [`2.2 模型架构`](./02_PyTorch_Algorithms/intro.md) / [`2.3 训练与微调闭环`](./02_PyTorch_Algorithms/intro.md) / [`2.4 偏好优化与对齐`](./02_PyTorch_Algorithms/intro.md) / [`2.5 反向传播与显存优化`](./02_PyTorch_Algorithms/intro.md) / [`2.6 核心推理优化`](./02_PyTorch_Algorithms/intro.md) / [`2.7 高级推理策略`](./02_PyTorch_Algorithms/intro.md) / [`2.8 模型压缩与量化`](./02_PyTorch_Algorithms/intro.md) / [`2.9 分布式并行策略`](./02_PyTorch_Algorithms/intro.md) / [`2.10 项目实战`](./02_PyTorch_Algorithms/intro.md) | 在 PyTorch 层把算法、模型、推理、压缩、并行与项目验证先跑通。 | 希望先用熟悉工具建立实现感的人。 | ✅ 已完成，持续优化 |
| [`第三部分：Triton 算子开发（5 组 / 15 节，已完成，持续优化）`](./03_Triton_Kernels/intro.md) | [`3.1 基础篇（5 节）`](./03_Triton_Kernels/intro.md) / [`3.2 过渡篇（2 节）`](./03_Triton_Kernels/intro.md) / [`3.3 进阶A：Attention优化（3 节）`](./03_Triton_Kernels/intro.md) / [`3.4 进阶B：推理优化（2 节）`](./03_Triton_Kernels/intro.md) / [`3.5 项目篇（3 节）`](./03_Triton_Kernels/intro.md) | 把前面学到的算子和优化思路落到 GPU kernel。 | 希望从 PyTorch 走向 Triton 的学习者。 | ✅ 已完成，持续优化 |
| [`第四部分：CUDA C++ 与系统优化（4 组 / 16 节，建设中）`](./04_CUDA_and_System_Optimization/intro.md) | [`4.1 CUDA 编程基础（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.2 系统级性能优化（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.3 分布式训练工程（4 节）`](./04_CUDA_and_System_Optimization/intro.md) / [`4.4 架构视野（4 节）`](./04_CUDA_and_System_Optimization/intro.md) | 进一步下探到 CUDA、系统调优和工程化架构。 | 准备做底层性能优化和工程落地的人。 | 🛠 建设中 |
| [`第五部分：CUDA Rust（预留）`](./05_CUDA_Rust/intro.md) | 预留中 | 预留中 | 预留中 | 🚧 预留 |

### 🧭 专题总览

| 层级 | 入口 | 覆盖范围 | 内容定位 | 适合对象 |
| ---- | ---- | ---- | ---- | ---- |
| 主学习路线 | [`监督微调专题`](./topic_discussion/fine_tuning_training/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/fine_tuning_training/intro.md)；正文：[casebook](./topic_discussion/fine_tuning_training/casebook.md)。SFT、LoRA、训练控制和项目交付。 | 想从 SFT 一路走到 LoRA 项目闭环的学习者。 |
| 主学习路线 | [`推理优化专题`](./topic_discussion/inference_optimization/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/inference_optimization/intro.md)；正文：[casebook](./topic_discussion/inference_optimization/casebook.md)。FlashAttention、解码、PagedAttention、cache 与 benchmark。 | 想系统理解推理加速路径的学习者。 |
| 主学习路线 | [`显存优化专题`](./topic_discussion/memory_performance_tuning/intro.md) | Part 0-2 | 导读：[intro](./topic_discussion/memory_performance_tuning/intro.md)；正文：[casebook](./topic_discussion/memory_performance_tuning/casebook.md)。VRAM、activation、checkpointing、offload 和 trade-off。 | 想系统优化显存和端到端性能的学习者。 |
| 横切支撑专题 | [`量化与压缩专题`](./topic_discussion/quantization/intro.md) | Part 0-3 | 导读：[intro](./topic_discussion/quantization/intro.md)；正文：[casebook](./topic_discussion/quantization/casebook.md)。PTQ、QAT、GPTQ、AWQ、FP8 与部署决策。 | 想同时考虑精度、显存、吞吐和部署取舍的学习者。 |
| 横切支撑专题 | [`通信与并行专题`](./topic_discussion/communication_parallel/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/communication_parallel/intro.md)；正文：[casebook](./topic_discussion/communication_parallel/casebook.md)。NCCL、AllReduce、ZeRO、PP、TP 和并行验证。 | 想理解多卡训练和通信边界的学习者。 |
| 横切支撑专题 | [`Profiling 专题`](./topic_discussion/profiling/intro.md) | Part 0-2 | 导读：[intro](./topic_discussion/profiling/intro.md)；正文：[casebook](./topic_discussion/profiling/casebook.md)。性能取证、trace 阅读、回归验证和行动决策。 | 想系统补性能意识与排障方法的学习者。 |
| 横切支撑专题 | [`后训练与对齐专题`](./topic_discussion/post_training_alignment/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/post_training_alignment/intro.md)；正文：[casebook](./topic_discussion/post_training_alignment/casebook.md)。RLHF、DPO、GRPO、偏好数据与项目收口。 | 想从 SFT 继续走到偏好优化与对齐的学习者。 |
| 基础支撑专题 | [`反向传播与训练机制专题`](./topic_discussion/backpropagation_training_mechanism/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/backpropagation_training_mechanism/intro.md)；正文：[casebook](./topic_discussion/backpropagation_training_mechanism/casebook.md)。autograd、backward、checkpointing、offload 与训练节奏。 | 想补训练机制底座的学习者。 |
| 基础支撑专题 | [`大模型架构专题`](./topic_discussion/model_architecture/intro.md) | Part 1-2 | 导读：[intro](./topic_discussion/model_architecture/intro.md)；正文：[casebook](./topic_discussion/model_architecture/casebook.md)。结构演进、代表模型和 MoE / 稀疏化。 | 想补模型结构背景与横向对照的学习者。 |
| 基础支撑专题 | [`编译与图优化专题`](./topic_discussion/compiler_graph_optimization/intro.md) | Part 1-4 | 导读：[intro](./topic_discussion/compiler_graph_optimization/intro.md)；正文：[casebook](./topic_discussion/compiler_graph_optimization/casebook.md)。图优化、fusion、lowering、schedule 和 backend 约束。 | 想理解图级优化与编译链路的学习者。 |

### 🤝 共学沉淀

| 模块 | 覆盖范围 | 内容定位 | 适合对象 | 状态 |
| ---- | ---- | ---- | ---- | ---- |
| [`组队学习专题`](./team_study/intro.md) | 不固定 | [`part2_l1_202606`](./team_study/part2_l1_202606/intro.md) / [`part2_l1_202607`](./team_study/part2_l1_202607/intro.md) / [`part2_l2_202607`](./team_study/part2_l2_202607/intro.md) | 想通过共学沉淀知识、题目与复盘记录的学习者。 | 🛠 建设中 |

## 🆕 更新时间线

- **2026-07-10**：[最新更新点]收紧了中文版首页的教材总览与状态列，校正了 `Part 0` / `Part 1` 的组名、节数和 `0E` 标题，并同步了相关导航与最近更新说明。
- **2026-06-26**：[最新更新点]收紧了中文版首页的教材总览、状态列和 mermaid 关系图，明确了 `Part 0-1` 的前置关系、`Part 2-5` 的主线关系，以及横向专题和组队学习的定位。
- **2026-06-15**：推进第零部分 / 第一部分的分组与导读收口，统一部分级导航，并完成网页底部评论区接入 GitHub Discussions，同时持续扩展第一部分的正文、桥接页与 Notebook 结构。
- **2026-06-13**：修复 dead link，并为未完成页面补充占位页，避免学习入口出现 404。
- **2026-04-21**：更新 Colab 徽章链接，统一指向官方 `datawhalechina` 仓库。
- **2026-04-20**：上线站点首页与部分导学；新增第零部分前置知识与第一部分练习内容，完善在线阅读入口与学习路径。
- **2026-04-18 ~ 2026-04-19**：集中重构第二部分 / 第三部分内容，优化 Notebook、答案区与算子实现说明。
- **2026-04-02**：完成教程核心 Notebook、文档与测试脚本的初始搭建。

> 路径兼容说明：第三部分已从 `03_CUDA_and_Triton_Kernels` 更名为 `03_Triton_Kernels`，CUDA / 系统优化内容拆分到第四部分。旧网页路径会保留迁移入口，建议新链接统一使用 `03_Triton_Kernels`。
## 🚀 快速开始

如果你想开始学习，不需要从 `00` 按顺序起步；在线站点的导学和目录是入口，不是硬性起点。Part 0 适合补基础，Part 1 / 2 / 3 / 4 可以按你的目标直接切入。需要运行 Notebook 时，Part 0 / 1 / 2 可以优先走 CPU-first，Part 3 / 4 需要 GPU 环境。环境与平台差异见 [使用指南](./docs/guide.md)。

### 学习路径

1. 在左侧侧边栏选择你当前最关心的部分
2. 点击 **📖 完整导学** 了解该部分的阅读顺序
3. 直接从对应 group 进入，不必先补完全部前置
4. 如果后面遇到知识缺口，再回到 Part 0 / Part 1 补基础
5. 环境和平台差异见 [使用指南](./docs/guide.md)

### 方式 1：在线阅读

访问在线站点：

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

适合：
- 先看目录再决定从哪一部分切入
- 先读部分导学，按目标跳转到对应 group
- Part 0 / 1 / 2 可以直接用 Colab CPU 跑练习
- Part 3 / 4 需要 Colab GPU runtime

### 方式 2：本地学习

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

适合：
- 想在本地完整跑 Part 0 / 1 / 2 的 Notebook
- 想自己控制 Python / PyTorch / CUDA 版本
- 想做更稳定的离线调试
- Part 3 / 4 需要本地 NVIDIA GPU

### 方式 3：CNB 统一环境

如果你希望和仓库当前推荐环境保持一致，可以使用 CNB 统一环境入口。

适合：
- 团队协作
- 统一实验镜像
- 需要减少本地环境差异
- Part 0 / 1 / 2 可以用 CNB CPU
- Part 3 / 4 需要 CNB GPU 会话

CNB 的具体使用方式和适用范围见 [使用指南](./docs/guide.md)。

## 📖 更多资源

- [使用指南](./docs/guide.md) - 环境与学习方式
- [贡献指南](./docs/contributing.md) - 如何参与项目开发和测试
- [维护与发布手册](./docs/maintenance.md) - 部分、链接、测试与发布的维护约定
- [自动化测试脚本索引](./docs/maintenance.md#测试脚本索引) - 各类验证脚本入口

## 👨‍💻 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| lynn_jingjing | 项目发起人 | 一个算法工程师 |


## 📄 许可声明

本仓库所有 `.ipynb` 文件中的文字内容（Markdown 单元格、公式、图示说明）采用 CC BY 4.0 协议；代码内容（Code 单元格、可执行实现）采用 Apache-2.0 协议。使用、转载、改编时，请按单元格类型分别遵守对应协议。文字协议见 [`LICENSE`](./LICENSE)，代码协议见 [`LICENSE-CODE`](./LICENSE-CODE)。

---

# English Version

## 📄 License Notice

All `.ipynb` files in this repository are mixed-content notebooks: Markdown cells (tutorial text, formulas, and figure captions) are licensed under CC BY 4.0, while Code cells (executable implementations) are licensed under Apache-2.0. Please comply with the corresponding license by cell type when using, redistributing, or adapting this repository. See [`LICENSE`](./LICENSE) for text and [`LICENSE-CODE`](./LICENSE-CODE) for code.

## 🎯 Project Introduction

This is a practical LLM algorithm tutorial from beginner to advanced, built around runnable, verifiable notebooks that help you move from "reading" to "writing, debugging, and optimizing".

### ✨ Features

1. **Clear Main Line**: A complete learning chain from prerequisites to Triton / CUDA system optimization.
2. **Engineering-Oriented**: Notebook-based practice with hands-on implementation and performance awareness.
3. **Broad Coverage**: Covers PyTorch, Transformers, inference optimization, VRAM management, and low-level implementation.

### 👥 Suitable For

- **Job Seekers**: Reinforce common interview topics for LLM algorithm engineers, AI architects, and kernel developers.
- **AI Practitioners**: Understand VRAM optimization, distributed communication, and Triton/CUDA operators from the code level.


## 🌐 Tutorial Overview

This tutorial is organized into four layers: the vertical main line, route-oriented study paths, cross-cutting topics, and collaborative study. `Part 0 -> Part 4` remains the main line, `topic_discussion` reorganizes training, inference, memory, quantization, parallelism, and profiling into navigable topic paths, and `team_study` is maintained as a separate collaborative-learning lane. The overview is summarized in the asset and topic tables below.

![Tutorial overview fallback](./docs/image-1.png)


### 📚 Current Asset Overview

You do not need to start from `00` in strict order. `00` is the prerequisite lane; if you already have the background, jump directly to the part that matches your goal. The table below summarizes each part, its groups, its audience, and its status.

| Part | Groups | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| [部分导读：前置知识与环境准备（5 groups / 20 lessons）](./00_Prerequisites/intro.md) | [组内导读：0A Python Basics and Data Representation (4 lessons)](./00_Prerequisites/0A.md) / [组内导读：0B PyTorch Tensors and Autograd (4 lessons)](./00_Prerequisites/0B.md) / [组内导读：0C PyTorch Model Construction (4 lessons)](./00_Prerequisites/0C.md) / [组内导读：0D Training and Model Intuition (4 lessons)](./00_Prerequisites/0D.md) / [组内导读：0E Debugging and Performance (4 lessons)](./00_Prerequisites/0E.md) | Prerequisites, engineering basics, and notebook-first practice. | First-time learners who need prerequisite support. | ✅ Complete, continuously refining |
| [部分导读：硬件、数学与系统（5 groups / 33 lessons）](./01_Hardware_Math_and_Systems/intro.md) | [组内导读：1A Numerics and Compute Estimation (4 lessons)](./01_Hardware_Math_and_Systems/1A.md) / [组内导读：1B Single-GPU Memory and Access (5 lessons)](./01_Hardware_Math_and_Systems/1B.md) / [组内导读：1C Multi-GPU Communication and VRAM (5 lessons)](./01_Hardware_Math_and_Systems/1C.md) / [组内导读：1D Heterogeneous Scheduling and Operators (5 lessons)](./01_Hardware_Math_and_Systems/1D.md) / [组内导读：1E Compiler Optimization and Hardware Ecosystem (4 lessons)](./01_Hardware_Math_and_Systems/1E.md) | Hardware, compute estimation, memory access, communication, and scheduling constraints. | Learners who want to understand why things are written and deployed this way. | ✅ Complete, continuously refining |
| [部分导读：PyTorch 算法实战（10 groups）](./02_PyTorch_Algorithms/intro.md) | [组内导读：2.1 Basic Operators](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.2 Model Architecture](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.3 Training and Fine-Tuning Loop](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.4 Preference Optimization and Alignment](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.5 Backpropagation and VRAM Optimization](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.6 Core Inference Optimization](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.7 Advanced Inference Strategies](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.8 Model Compression and Quantization](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.9 Distributed Parallel Strategy](./02_PyTorch_Algorithms/intro.md) / [组内导读：2.10 Projects](./02_PyTorch_Algorithms/intro.md) | PyTorch-level practice for algorithms, models, inference, compression, parallelism, and project validation. | Learners who want to build implementation intuition with familiar tools. | ✅ Complete, continuously refining |
| [部分导读：Triton Kernel Development (5 groups / 15 lessons)](./03_Triton_Kernels/intro.md) | [组内导读：3.1 Foundations (5 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.2 Transition (2 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.3 Advanced A: Attention Optimization (3 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.4 Advanced B: Inference Optimization (2 lessons)](./03_Triton_Kernels/intro.md) / [组内导读：3.5 Projects (3 lessons)](./03_Triton_Kernels/intro.md) | Triton kernel development. | Learners who want to move from PyTorch to Triton. | ✅ Complete, continuously refining |
| [Part 4: CUDA C++ and System Optimization (4 groups / 16 lessons)](./04_CUDA_and_System_Optimization/intro.md) | [4.1 CUDA Programming Basics (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.2 System-Level Performance Optimization (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.3 Distributed Training Engineering (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) / [4.4 Architecture Perspective (4 lessons)](./04_CUDA_and_System_Optimization/intro.md) | CUDA C++ and system optimization. | Learners preparing for low-level performance optimization and engineering deployment. | 🛠 In progress |
| [Part 5: CUDA Rust (reserved)](./05_CUDA_Rust/intro.md) | Reserved | Reserved | Reserved | 🚧 Reserved |

### 🧭 Topic Overview

| Layer | Entry | Coverage | Content Positioning | Suitable For |
| ---- | ---- | ---- | ---- | ---- |
| Main Study Path | [Fine-Tuning Training Topic](./topic_discussion/fine_tuning_training/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/fine_tuning_training/intro.md); casebook: [casebook](./topic_discussion/fine_tuning_training/casebook.md). SFT, LoRA, training control, and project delivery. | Learners who want to go from SFT to a LoRA project closure. |
| Main Study Path | [Inference Optimization Topic](./topic_discussion/inference_optimization/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/inference_optimization/intro.md); casebook: [casebook](./topic_discussion/inference_optimization/casebook.md). FlashAttention, decoding, PagedAttention, cache, and benchmark. | Learners who want practical inference acceleration. |
| Main Study Path | [Memory and Performance Tuning Topic](./topic_discussion/memory_performance_tuning/intro.md) | Part 0-2 | Guide: [intro](./topic_discussion/memory_performance_tuning/intro.md); casebook: [casebook](./topic_discussion/memory_performance_tuning/casebook.md). VRAM, activation, checkpointing, offload, and trade-offs. | Learners who want to optimize memory usage and end-to-end performance. |
| Cross-Cutting Topic | [Quantization Topic](./topic_discussion/quantization/intro.md) | Part 0-3 | Guide: [intro](./topic_discussion/quantization/intro.md); casebook: [casebook](./topic_discussion/quantization/casebook.md). PTQ, QAT, GPTQ, AWQ, FP8, and deployment decisions. | Learners balancing accuracy, memory, throughput, and deployment cost. |
| Cross-Cutting Topic | [Communication and Parallelism Topic](./topic_discussion/communication_parallel/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/communication_parallel/intro.md); casebook: [casebook](./topic_discussion/communication_parallel/casebook.md). NCCL, AllReduce, ZeRO, PP, TP, and validation. | Learners who want to understand multi-GPU scaling and communication cost. |
| Cross-Cutting Topic | [Profiling Topic](./topic_discussion/profiling/intro.md) | Part 0-2 | Guide: [intro](./topic_discussion/profiling/intro.md); casebook: [casebook](./topic_discussion/profiling/casebook.md). Evidence collection, trace reading, regression validation, and action decisions. | Learners who want systematic performance diagnosis and debugging methods. |
| Cross-Cutting Topic | [Post-Training Alignment Topic](./topic_discussion/post_training_alignment/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/post_training_alignment/intro.md); casebook: [casebook](./topic_discussion/post_training_alignment/casebook.md). RLHF, DPO, GRPO, preference data, and project closure. | Learners who want to continue from SFT into alignment and preference optimization. |
| Foundation Topic | [Backpropagation and Training Mechanics Topic](./topic_discussion/backpropagation_training_mechanism/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/backpropagation_training_mechanism/intro.md); casebook: [casebook](./topic_discussion/backpropagation_training_mechanism/casebook.md). Autograd, backward, checkpointing, offload, and training rhythm. | Learners who want stronger training-mechanism foundations. |
| Foundation Topic | [Model Architecture Topic](./topic_discussion/model_architecture/intro.md) | Part 1-2 | Guide: [intro](./topic_discussion/model_architecture/intro.md); casebook: [casebook](./topic_discussion/model_architecture/casebook.md). Structure evolution, representative models, and MoE/sparsity. | Learners who want structural background and model comparison. |
| Foundation Topic | [Compiler and Graph Optimization Topic](./topic_discussion/compiler_graph_optimization/intro.md) | Part 1-4 | Guide: [intro](./topic_discussion/compiler_graph_optimization/intro.md); casebook: [casebook](./topic_discussion/compiler_graph_optimization/casebook.md). Graph optimization, fusion, lowering, schedule, and backend constraints. | Learners who want compiler and graph-level optimization vision. |

### 🤝 Collaborative Study

| Module | Coverage | Content Positioning | Suitable For | Status |
| ---- | ---- | ---- | ---- | ---- |
| [Team Study Topic](./team_study/intro.md) | Not fixed | [part2_l1_202606](./team_study/part2_l1_202606/intro.md) / [part2_l1_202607](./team_study/part2_l1_202607/intro.md) / [part2_l2_202607](./team_study/part2_l2_202607/intro.md) | Learners who want to accumulate knowledge and review records through collaborative study. | 🛠 In progress |

## 🆕 Update Timeline

- **2026-07-10**: [Latest update] tightened the English homepage asset overview and status columns, aligned the part/group counts with the current source structure, and refreshed the topic and team-study status tables.
- **2026-06-26**: [Latest update] improved the Chinese homepage overview and clarified the learning path across Parts 3 and 4, making the entry points and study order more intuitive.
- **2026-06-15**: Finalized the Part 0 / 1 grouping and guide cleanup, unified the part-level navigation, connected the page comments to GitHub Discussions, and continued expanding Part 1 content, bridge pages, and notebook structure.
- **2026-06-13**: Fixed dead links and added placeholder pages for unfinished content to prevent 404s in learning entry points.
- **2026-04-21**: Updated Colab badges to point to the official `datawhalechina` repository.
- **2026-04-20**: Launched the site homepage and part guides; added Part 0 prerequisites and Part 1 practice content to unify the learning path.
- **2026-04-18 ~ 2026-04-19**: Refactored Part 2 / 3 content, polishing notebooks, answer sections, and operator implementation notes.
- **2026-04-02**: Completed the initial tutorial notebooks, docs, and test scripts.

> Path compatibility note: Part 3 has been renamed from `03_CUDA_and_Triton_Kernels` to `03_Triton_Kernels`, and CUDA / system optimization content has moved to Part 4. Old web paths keep migration pages, but new links should use `03_Triton_Kernels`.

## 🚀 Quick Start

You do not need to start from Part 0 in order; Part 0 is the prerequisite lane, and you can jump directly to the part that matches your goal.

### Option 1: Read Online

Visit the online platform:

[https://datawhalechina.github.io/llm-algo-leetcode/](https://datawhalechina.github.io/llm-algo-leetcode/)

Suitable for:
- Skimming the table of contents first and then jumping to the part you need
- Reading the part guides first
- Part 0 / 1 / 2 can run on Colab CPU
- Part 3 / 4 need a Colab GPU runtime

### Option 2: Local Development

```bash
git clone https://github.com/datawhalechina/llm-algo-leetcode.git
cd llm-algo-leetcode
conda env create -f environment.yml
conda activate llm_algo
jupyter lab
```

Suitable for:
- Running Part 0 / 1 / 2 locally on CPU
- Controlling your own Python / PyTorch / CUDA versions
- More stable offline debugging
- Part 3 / 4 require a local NVIDIA GPU

For environment details and platform differences, see the Chinese guide section or [docs/guide.md](./docs/guide.md).

### Option 3: CNB Unified Delivery

If you want the same runtime style used by the repository, use the CNB unified environment.

Suitable for:
- Team collaboration
- Consistent experiment images
- Lower local environment drift
- Part 0 / 1 / 2 can use CNB CPU
- Part 3 / 4 need a CNB GPU session

See [docs/guide.md](./docs/guide.md) for the exact environment rules and scope.

## 📖 More Resources

- [docs/guide.md](./docs/guide.md) - environment and learning modes
- [docs/contributing.md](./docs/contributing.md) - how to contribute to development and testing
- [docs/maintenance.md](./docs/maintenance.md) - maintenance rules for parts, links, tests, and releases
- [Automated Test Script Index](./docs/maintenance.md#测试脚本索引) - entry points for automated verification scripts

## 👨‍💻 Contributors

| Name | Role | Description |
| :---- | :---- | :---- |
| lynn_jingjing | Project initiator | An algorithm engineer |

*(Feel free to add your name here! )*

## 📄 License

Tutorial text in this repository is licensed under [CC BY 4.0](./LICENSE), and code is licensed under [Apache-2.0](./LICENSE-CODE). `.ipynb` files are mixed-content notebooks, so please follow the corresponding license by cell type.
