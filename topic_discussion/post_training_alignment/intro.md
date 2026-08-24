# 后训练与对齐专题

## 专题定位与 Infra 层定位

本专题串起 SFT 之后的后训练主线：先看为什么模型在 SFT 之后还会出现偏好错位，再看 PPO、DPO、GRPO 分别改的是哪一层问题，最后把偏好数据、评测和项目结论收成同一条对齐闭环。后训练以 Infra-L3 的损失、采样、reference model 和训练运行时为核心；在线采样、推理服务和反馈回路才延伸到 Infra-L4，Infra-L5 负责数据生命周期、实验编排、评测回归和交付治理。

Infra-L1/Infra-L2 的显存、算子和通信成本会直接影响 PPO、DPO、GRPO 的可行性，因此最终结论不能只看对齐指标，还要记录吞吐、显存、稳定性和服务代价。若还没有 SFT 基础，应先回监督微调专题；若重点转为单模型推理或显存预算，应转到对应专题。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的后训练与对齐路线进入，先完成监督微调基础，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 84、85、86 三个项目节。本专题适合按问题选择 DPO、GRPO 或在线流程，不要求先完整学习 PPO。

## 前置阅读

建议先掌握监督微调专题中的数据、训练循环和 LoRA 基础，再阅读 `Part 02: 2.4` 的偏好数据与对齐内容。进入项目前至少明确 preference pair / group 数据格式、reference model、奖励或偏好指标，以及训练和评测的隔离方式。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 为什么 SFT 之后还要对齐 | `14` | [01 Why Post-Training Alignment](./01_why_post_training_alignment.md) |
| Task2 | RLHF / PPO 的系统代价 | `14` | [02 RLHF and PPO System Cost](./02_rlhf_and_ppo_system_cost.md) |
| Task3 | DPO 与偏好优化 | `15 -> 84 -> 86` | [03 DPO and Preference Optimization](./03_dpo_and_preference_optimization.md) |
| Task4 | GRPO 与 group-wise 对齐 | `16 -> 85` | [04 GRPO and Groupwise Alignment](./04_grpo_and_groupwise_alignment.md) |
| Task5 | 偏好数据、在线优化与冲突评测 | `50 -> 51 -> 52` | [05 Preference Data and Evaluation](./05_preference_data_and_evaluation.md) |
| Task6 | 项目收口与采用建议 | `84 -> 85 -> 86` | [06 Project Decision and Delivery](./06_project_decision_and_delivery.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“为什么 SFT 还不够”“PPO / DPO / GRPO 到底差在哪一层”时，再回来看对应的专题正文。想看汇总版就进 [后训练与对齐正文](./casebook.md)，想按连续故事线走一遍就进 [后训练与对齐深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[监督微调专题](../fine_tuning_training/intro.md) 负责 SFT 前置，[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责 alignment loss 的 backward 代价，[显存优化专题](../memory_performance_tuning/intro.md) 负责 PPO 等系统成本。

## 项目结论

推荐的实践闭环是 `84 DPO 偏好项目 -> 85 GRPO 组内对齐项目 -> 86 在线 DPO benchmark`。学习者最终应比较数据质量、训练稳定性、偏好或任务评测、显存与吞吐，而不是只比较训练 loss；在线项目还要额外记录采样、打分和更新链路的成本。

## 环境与验证

损失函数、数据格式和小规模离线验证可先用 CPU；完整 SFT、DPO、GRPO 训练通常需要 GPU，在线 benchmark 还需要可用的推理后端或服务接口。应固定数据切分、随机种子、reference 配置和评测集，并保存训练配置、指标曲线和最终决策。
