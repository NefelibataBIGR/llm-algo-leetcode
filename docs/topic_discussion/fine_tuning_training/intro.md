# 监督微调专题

## 专题定位

本专题用于串起训练微调主线：先把模型结构和 PyTorch 基础接到 SFT / LoRA，再进入训练控制、数据工程、readiness 和项目交付。这里聚焦“结构前置 -> SFT / LoRA -> 训练控制 -> 项目收口”；如果问题已经转到偏好优化或对齐算法，应转到后训练与对齐专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part00 / Part01 / Part02` 的具体小节；最后一列主要对应 `01-05` 的专题正文页，分支补充放在工程与交付附录。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | PyTorch 与模型结构入口 | `00 -> 01 -> 02` | [01 SFT Data and Loss](./01_sft_data_and_loss.md) |
| Task2 | 位置编码、注意力与 block 前置 | `03 -> 04 -> 05 -> 08` | [02 LoRA PEFT Design](./02_lora_peft_design.md) |
| Task3 | SFT 与 LoRA 闭环 | `09 -> 10` | [02 LoRA PEFT Design](./02_lora_peft_design.md) |
| Task4 | 训练控制与端到端实验 | `11 -> 12 -> 13` | [03 Training Control](./03_training_control.md) |
| Task5 | 数据工程、readiness 与项目交付 | `32 -> 33 -> 60` | [05 Project Delivery and Decision](./05_project_delivery_decision.md) |
| Task6 | 小显存、LoRA 变体与后续分叉 | `26 -> 31 -> 15 -> 16` | [训练工程附录](./training_engineering_appendix.md), [项目交付附录](./project_delivery_appendix.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“LoRA 为什么挂在这些层上”“训练控制和项目交付怎么接起来”时，再回来看对应的专题正文。想看汇总版就进 [监督微调正文](./casebook.md)，想按连续故事线走一遍就进 [监督微调深入阅读](./walkthrough.md)。工具层补充放在 [训练工程附录](./training_engineering_appendix.md)，项目证据链补充放在 [项目交付附录](./project_delivery_appendix.md)。

如果问题已经跨到别的专题：
[大模型架构专题](../model_architecture/intro.md) 负责结构地基，[后训练与对齐专题](../post_training_alignment/intro.md) 负责 DPO / GRPO 分叉，[量化与压缩专题](../quantization/intro.md) 负责 QLoRA 与压缩路线，[显存优化专题](../memory_performance_tuning/intro.md) 负责 OOM 与显存账本，[Profiling 专题](../profiling/intro.md) 负责训练证据链，[通信与并行专题](../communication_parallel/intro.md) 负责多卡边界。
