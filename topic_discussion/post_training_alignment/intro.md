# 后训练与对齐专题

## 专题定位

本专题用于串起 SFT 之后的后训练主线：先看为什么模型在 SFT 之后还会出现偏好错位，再看 PPO、DPO、GRPO 分别改的是哪一层问题，最后把偏好数据、评测和项目结论收成同一条对齐闭环。这里聚焦 SFT 之后的方法分化；如果还没有 SFT 基础，应先回监督微调专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 为什么 SFT 之后还要对齐 | `14` | [01 Why Post-Training Alignment](./01_why_post_training_alignment.md) |
| Task2 | RLHF / PPO 的系统代价 | `14` | [02 RLHF and PPO System Cost](./02_rlhf_and_ppo_system_cost.md) |
| Task3 | DPO 与偏好优化 | `15 -> 84 -> 86` | [03 DPO and Preference Optimization](./03_dpo_and_preference_optimization.md) |
| Task4 | GRPO 与 group-wise 对齐 | `16 -> 85` | [04 GRPO and Groupwise Alignment](./04_grpo_and_groupwise_alignment.md) |
| Task5 | 偏好数据与评测 | `50` | [05 Preference Data and Evaluation](./05_preference_data_and_evaluation.md) |
| Task6 | 项目收口与采用建议 | `84 -> 85 -> 86` | [06 Project Decision and Delivery](./06_project_decision_and_delivery.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“为什么 SFT 还不够”“PPO / DPO / GRPO 到底差在哪一层”时，再回来看对应的专题正文。想看汇总版就进 [后训练与对齐正文](./casebook.md)，想按连续故事线走一遍就进 [后训练与对齐深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[监督微调专题](../fine_tuning_training/intro.md) 负责 SFT 前置，[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责 alignment loss 的 backward 代价，[显存优化专题](../memory_performance_tuning/intro.md) 负责 PPO 等系统成本。
