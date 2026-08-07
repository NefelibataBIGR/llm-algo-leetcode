# 后训练与对齐专题

## 专题概览

本专题用于把 `Part 02` 里分散的后训练、偏好优化和对齐评测内容串成一条独立路线，回答“为什么 SFT 之后还要继续做对齐、DPO / GRPO / PPO 各自改什么、偏好数据和评测怎么接上项目闭环”。

这条线覆盖 `14-16 + 50 + 84-85`：先看 RLHF / PPO、DPO、GRPO 的机制边界，再看偏好数据和对齐评测，最后把 DPO / GRPO 落到项目页里形成可验证闭环。

## 职责边界

这个专题只负责后训练、偏好优化和对齐评测，不负责基础 SFT 主线，也不负责推理优化或显存调优。

- `RLHF / PPO` 关注对齐训练的系统代价、训练流转和四模型链路。
- `DPO` 关注轻量偏好优化的目标函数与数据对。
- `GRPO` 关注组内比较、group-wise 优势和更适合生成场景的对齐方式。
- `Preference Data / Eval` 关注 chosen / rejected、win-rate 和评测口径。
- `Project Pages` 关注把方法落成可交付的 DPO / GRPO 项目闭环。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1A / 1C` | 对齐训练前需要理解的规模估算、多卡通信和资源边界 |
| `Part 2.4` | RLHF / PPO、DPO、GRPO 的机制原理 |
| `Part 2.4` | 偏好数据、评测口径和对齐方法边界 |
| `Part 2.9` | DPO / GRPO 项目页，输出对齐闭环和决策 |

## Part 1 相关前置

- [1A](../../01_Hardware_Math_and_Systems/1A.md)：先看数量级、参数量和资源账本，知道对齐训练为什么会比普通 SFT 更重。
- [1C](../../01_Hardware_Math_and_Systems/1C.md)：先看多卡通信和显存共享，理解 RLHF / PPO 这类方法为什么容易碰到系统边界。

## Task1-6 路线

| Task | 内容 | 章节 |
|:---|:---|:---|
| Task1 | 对齐问题与方法谱系 | [14 RLHF PPO Memory](../../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.ipynb)、[15 DPO Loss Tutorial](../../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.ipynb)、[16 GRPO Loss Tutorial](../../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb) |
| Task2 | 偏好数据与对齐评测 | [50 Preference Data and Evaluation](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) |
| Task3 | DPO 项目闭环 | [84 DPO Preference Project](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb) |
| Task4 | GRPO 项目闭环 | [85 GRPO Groupwise Alignment Project](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb) |
| Task5 | 和训练主线 / 结构主线的联动 | [09 SFT Training Loop](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)、[10 LoRA Tutorial](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.ipynb)、[05 LLaMA3 Block Tutorial](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) |
| Task6 | 路线收口与项目决策 | [casebook.md](./casebook.md)、[walkthrough.md](./walkthrough.md) |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `14` | RLHF / PPO 的训练流转和系统代价 | [14 RLHF PPO Memory](../../02_PyTorch_Algorithms/14_RLHF_PPO_Memory.ipynb) |
| `15` | DPO 的目标函数、偏好数据和轻量对齐直觉 | [15 DPO Loss Tutorial](../../02_PyTorch_Algorithms/15_DPO_Loss_Tutorial.ipynb) |
| `16` | GRPO 的组内比较、group-wise 优势和稳定性 | [16 GRPO Loss Tutorial](../../02_PyTorch_Algorithms/16_GRPO_Loss_Tutorial.ipynb) |
| `50` | chosen / rejected、win-rate 和对齐评测口径 | [50 Preference Data and Evaluation](../../02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb) |
| `84` | DPO 项目页，输出偏好优化的项目结论 | [84 DPO Preference Project](../../02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb) |
| `85` | GRPO 项目页，输出组内对齐的项目结论 | [85 GRPO Groupwise Alignment Project](../../02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb) |

## 推荐入口

- 如果你第一次接触对齐，先看 `14 -> 15 -> 16`，把方法谱系和边界立住。
- 如果你想理解偏好数据和评测，接着看 `50`。
- 如果你想看怎么落成项目，直接看 `84 -> 85`。
- 如果你已经有 SFT / LoRA 基础，再回头看这条线会更顺。

## 入口摘要

- 最短对齐路线：`14 -> 15 -> 16 -> 50 -> 84 -> 85`。
- 机制优先路线：`14 -> 15 -> 16`。
- 项目优先路线：`50 -> 84 -> 85`。

## 正文页

- [casebook.md](./casebook.md)：按“方法谱系 / 数据与评测 / 项目决策 / 常见误区”展开。
- [walkthrough.md](./walkthrough.md)：按一条偏好样本从数据到项目结果的故事线展开。

## 相关专题

- [训练微调闭环专题](../fine_tuning_training/intro.md)：当你需要先把 SFT / LoRA 主线立住时先看这里。
- [反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md)：当你需要理解 loss、backward 和训练调度时先看这里。
- [大模型结构和原理专题](../model_architecture/intro.md)：当你需要理解后训练会触达哪些层时先看这里。

## 读法建议

- `14 / 15 / 16` 一起看，先把方法谱系和边界立住。
- `50` 用来补偏好数据与评测口径。
- `84 / 85` 用来把方法变成项目闭环。
- `09 / 10 / 05` 只是联动参考，不要替代对齐主线。

## 专题状态

当前为专题入口页，后续将逐步补充更完整的案例、评测口径和项目导流。
