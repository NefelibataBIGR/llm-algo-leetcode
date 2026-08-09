# 后训练与对齐专题

## 专题概览

本专题用于把 `Part 02` 里分散的**后训练、偏好优化和对齐评测**重新组织成一条完整故事线，回答三个核心问题：

- 为什么模型在 SFT 之后还要继续做对齐？
- PPO / DPO / GRPO 分别改的是哪一层问题，代价为什么不同？
- 偏好数据、评测口径和项目交付，怎样一起构成“对齐闭环”？

这条线承接 `Part 02` 已有学习路线，但不复述目录。`14 / 15 / 16 / 50 / 84 / 85` 只是素材来源；横向专题负责把这些素材串成“目标变化 -> 方法分化 -> 数据评测 -> 项目决策”的知识骨架。

## 职责边界

这个专题只负责**SFT 之后的后训练与对齐**，不负责 SFT 基础流程，也不负责推理服务优化或显存调优。

- `01` 解释为什么 SFT 之后还会出现偏好错位，以及后训练要解决什么。
- `02` 解释 RLHF / PPO 为什么完整但重，它的系统代价从哪里来。
- `03` 解释 DPO 为什么更轻，它依赖什么样的偏好数据。
- `04` 解释 GRPO 为什么更适合 group-wise 比较和生成类场景。
- `05` 解释 chosen / rejected、group candidates 和评测口径怎样共同决定结果可信度。
- `06` 把方法、数据、评测和项目页汇总成 adopt / tune / reject 的决策。
- `07` 负责图册收口。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1A / 1C` | 资源预算、多卡通信、为什么 PPO 容易碰到系统边界 |
| `Part 2.3` | SFT / LoRA 作为后训练前置 |
| `Part 2.4` | RLHF / PPO、DPO、GRPO 的方法原理 |
| `Part 2.5` | backward、显存和训练调度在对齐阶段的系统代价 |
| `Part 2.9` | DPO / GRPO 项目页和最终采用建议 |

## Task1-6 路线

`Task1-6` 仍然保留为学习路径；`01-06` 是知识组织层。二者并存，但不要求一一对应。

| Task | 学习内容 | 章节 |
|:---|:---|:---|
| Task1 | 对齐问题与方法谱系入口 | [14 RLHF PPO Memory](../../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.md)、[15 DPO Loss Tutorial](../../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.md)、[16 GRPO Loss Tutorial](../../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.md) |
| Task2 | 偏好数据与评测口径 | [50 Preference Data and Evaluation](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.md) |
| Task3 | DPO 项目闭环 | [84 DPO Preference Project](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.md) |
| Task4 | GRPO 项目闭环 | [85 GRPO Groupwise Alignment Project](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.md) |
| Task5 | 与 SFT / 训练机制主线的联动 | [09 SFT Training Loop](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)、[10 LoRA Tutorial](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)、[17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.md)、[19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.md) |
| Task6 | 方法、数据、评测与项目决策收口 | [01 Why Post-Training](./01_why_post_training_alignment.md) 到 [07 Visual Assets](./07_visual_assets.md) |

## 01-06 骨架

这 6 个编号页是专题正文，不是文件索引。它们围绕“模型为什么要继续对齐、又该怎么落地”来组织。

| 章节 | 你会得到什么 | 适合先从哪里进入 |
|:---|:---|:---|
| `01` | SFT 之后为什么仍然会偏离偏好，后训练到底在修什么 | 先想弄清楚为什么还要做对齐 |
| `02` | RLHF / PPO 的完整闭环、四模型链和系统代价 | 先看经典路线或想理解为什么 PPO 重 |
| `03` | DPO 的目标改写、偏好对和轻量化边界 | 先看轻量偏好优化 |
| `04` | GRPO 的 group-wise 视角和生成类场景优势 | 先看候选组比较和排序式对齐 |
| `05` | 偏好数据、评测口径和常见失真点 | 先看 chosen / rejected 和 win-rate 怎么读 |
| `06` | 项目交付、benchmark 和 adopt / tune / reject 决策 | 已经有候选方法，想直接收口到项目页 |

## 方法主线

可以把这条专题先粗略理解成下面这条故事线：

```text
SFT model
  │
  ▼
alignment gap
  │
  ├─ RLHF / PPO: reward model + rollout + policy update
  ├─ DPO: preference pair + direct objective
  └─ GRPO: candidate groups + relative comparison
  │
  ▼
preference data / evaluation
  │
  ▼
project delivery
```

## 文献锚点

每个正文页都应该补 3-5 篇代表文献，至少覆盖以下主线：

- `02 RLHF / PPO`：InstructGPT、PPO 及其工程实践。
- `03 DPO`：DPO 原始论文及其后续变体。
- `04 GRPO`：group-wise / relative preference 相关工作。
- `05 Preference Data and Evaluation`：偏好数据清洗、judge 评测和 win-rate 使用边界。
- `06 Project Decision`：把方法与评测放回项目闭环时的采用标准。

## 推荐入口

- 如果你第一次接触对齐，先看 `01 -> 02 -> 03 -> 04`，先把方法谱系立住。
- 如果你已经知道方法名，但不知道数据和评测怎么接，直接看 `05`。
- 如果你最关心项目落地，直接看 `06`，再回跳 `03 / 04 / 05`。
- 如果你还没有 SFT 基础，先去 [监督微调（SFT）闭环专题](../fine_tuning_training/intro.md)。

## 正文页

- [01 Why Post-Training Alignment](./01_why_post_training_alignment.md)
- [02 RLHF and PPO System Cost](./02_rlhf_and_ppo_system_cost.md)
- [03 DPO and Preference Optimization](./03_dpo_and_preference_optimization.md)
- [04 GRPO and Groupwise Alignment](./04_grpo_and_groupwise_alignment.md)
- [05 Preference Data and Evaluation](./05_preference_data_and_evaluation.md)
- [06 Project Decision and Delivery](./06_project_decision_and_delivery.md)
- [07 Visual Assets](./07_visual_assets.md)
- [后训练与对齐正文](./casebook.md)
- [后训练与对齐深入阅读](./walkthrough.md)

## 相关专题

- [监督微调（SFT）闭环专题](../fine_tuning_training/intro.md)：当你还需要先把 SFT / LoRA 主线跑通时先看这里。
- [反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md)：当你需要理解 alignment loss、backward 和训练代价时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当你需要解释 PPO / rollout 为何更吃显存时看这里。
- [通信并行专题](../communication_parallel/intro.md)：当对齐训练扩到多卡、MoE 或专家并行时看这里。

## 专题状态

当前专题已收成 `01-06 + 07_visual_assets` 的正文骨架。下一步重点应放在：补文献锚点、补第一批图、把 `84 / 85` 的项目结论更明确地挂回 `06`。
