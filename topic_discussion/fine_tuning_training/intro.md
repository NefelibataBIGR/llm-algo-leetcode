# 监督微调专题

## 专题定位与 Infra 层定位

本专题串起训练微调主线：先把模型结构和 PyTorch 基础接到 SFT / LoRA，再进入训练控制、数据工程、readiness 和项目收口。监督微调主要落在 Infra-L3 框架与训练运行时，受 Infra-L1 的计算、显存和存储条件约束，也受 Infra-L2 算子库、通信库和编译结果影响；多卡训练与交付治理分别延伸到 Infra-L3/Infra-L5，服务化不是本专题的核心范围。

因此，LoRA、QLoRA 和梯度累积不能只当作独立技巧，应放回模型负载、显存预算、step time、吞吐、训练稳定性和评测结果中判断。若问题已经转到 PPO、DPO、GRPO 等偏好优化，应转到后训练与对齐专题；若问题变成单卡 kernel 或图融合瓶颈，应转到编译与图优化专题。

## 推荐入口

推荐从 Part 02 的 [2.3 训练与微调闭环](../../02_PyTorch_Algorithms/2_3.md) 开始；结构基础不足时回补 [2.1 基础算子](../../02_PyTorch_Algorithms/2_1.md) 和 [2.2 模型架构](../../02_PyTorch_Algorithms/2_2.md)。

## 前置阅读

- Part 00：Python、PyTorch 和训练循环基础。
- Part 01：GPU、显存和性能基本概念。
- Part 02：优先完成 2.1-2.3，再进入本专题的项目路线。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列主要对应 `01-05` 的专题正文页，分支补充放在工程与交付附录。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | PyTorch 与模型结构入口 | `00 -> 01 -> 02` | [01 SFT Data and Loss](./01_sft_data_and_loss.md) |
| Task2 | 位置编码、注意力与 block 前置 | `03 -> 04 -> 05 -> 08` | [02 LoRA PEFT Design](./02_lora_peft_design.md) |
| Task3 | SFT 与 LoRA 闭环 | `09 -> 10` | [02 LoRA PEFT Design](./02_lora_peft_design.md) |
| Task4 | 训练控制与端到端实验 | `11 -> 12 -> 13` | [03 Training Control](./03_training_control.md) |
| Task5 | 长上下文、数据工程、readiness 与项目交付 | `30 -> 32 -> 33 -> 60 -> 62 -> 63 -> 64` | [05 Project Delivery and Decision](./05_project_delivery_decision.md) |
| Task6 | 小显存、LoRA 变体与量化选型 | `26 -> 31 -> 65 -> 15 -> 16` | [训练工程附录](./training_engineering_appendix.md), [项目交付附录](./project_delivery_appendix.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“LoRA 为什么挂在这些层上”“训练控制和项目交付怎么接起来”时，再回来看对应的专题正文。想看汇总版就进 [监督微调正文](./casebook.md)，想按连续故事线走一遍就进 [监督微调深入阅读](./walkthrough.md)。工具层补充放在 [训练工程附录](./training_engineering_appendix.md)，项目证据链补充放在 [项目交付附录](./project_delivery_appendix.md)。

如果问题已经跨到别的专题：
[大模型架构专题](../model_architecture/intro.md) 负责结构地基，[后训练与对齐专题](../post_training_alignment/intro.md) 负责 DPO / GRPO 分叉，[量化与压缩专题](../quantization/intro.md) 负责 QLoRA 与压缩路线，[显存优化专题](../memory_performance_tuning/intro.md) 负责 OOM 与显存账本，[Profiling 专题](../profiling/intro.md) 负责训练证据链，[通信与并行专题](../communication_parallel/intro.md) 负责多卡边界。

## 项目结论

核心项目为 [60 LoRA 微调](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)、[62 指令微调](../../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb) 和 [63 LoRA 变体对比](../../02_PyTorch_Algorithms/63_LoRA_Variants_Benchmark.ipynb)；扩展项目包括 [64 数据质量](../../02_PyTorch_Algorithms/64_SFT_Data_Quality_Project.ipynb) 和 [65 QLoRA 选型](../../02_PyTorch_Algorithms/65_QLoRA_Selection_Project.ipynb)。

## 环境与验证

基础阅读和多数 Notebook 可 CPU-first；真实训练项目建议使用单 GPU。运行前先查看对应 Notebook 的环境说明，结果和结论以项目 Notebook 的输出为准。
