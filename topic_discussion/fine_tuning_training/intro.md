# 监督微调（SFT）闭环专题

## 专题概览

本专题用于把 `Part 02` 里**监督微调（SFT）的闭环**串成一条可执行路线，回答“怎么从 SFT 数据构造走到 LoRA 项目报告”。

它侧重微调，但仍然属于训练主线的一部分，不是训练以外的独立方向；这里不展开完整预训练，也不展开 RLHF / DPO，只聚焦 SFT / LoRA / 训练控制 / 项目交付这一段。

主线分成两层：

- **基础层**：`01-05 + 06`，先把 SFT 数据与 loss 对齐，再挂 LoRA / PEFT，接着统一训练控制、端到端实验和项目交付，最后在 `06` 用图册收口。
- **进阶层**：`30-32`，在基础闭环跑通之后，继续看长上下文微调、LoRA 变体和数据工程。

这样做的目的，是让学习者先跑通第一个可复现闭环，再进入更细的策略分化。基础层保留 6 页主结构，进阶层先占位不扩写。

## 职责边界

这个专题负责监督微调闭环和项目交付，不负责完整预训练、不负责对齐算法主线，也不负责推理服务优化。

- `01 SFT Data / Loss` 关注 `input_ids / attention_mask / labels` 和 response-only loss。
- `02 LoRA / PEFT` 关注 target modules、rank、alpha、dropout 和可训练参数账本。
- `03 Training Control` 关注 scheduler、effective batch 和 optimizer update 计数。
- `04 Experiment Report` 关注 train / val loss、显存、速度和项目决策。
- `05 Project Delivery` 关注 adapter、tokenizer、config、artifact 和采用建议。
- `06 Visual Assets` 负责把基础闭环的关键图收口。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 1A / 1B / 1C` | 训练前需要理解的规模估算、单卡显存和多卡通信前置 |
| `Part 2.1 / 2.2` | 训练微调前需要理解的模型结构地基 |
| `Part 2.3` | SFT、LoRA、scheduler、gradient accumulation、端到端实验 |
| `Part 2.9` | LoRA 微调项目收口 |
| `Part 2.4 / 2.7` | 长上下文微调、LoRA 变体、数据工程 |
| `Part 2.5 / 2.7B` | Autograd、backward、checkpointing、QLoRA 等增强选读 |

## Part 1 相关前置

- [1A](../../01_Hardware_Math_and_Systems/1A.md)：先看数量级、参数量和资源账本，知道训练大概会落在哪个规模。
- [1B](../../01_Hardware_Math_and_Systems/1B.md)：先看单卡硬件、访存和显存估算，避免一上来就只盯着 loss。
- [1C](../../01_Hardware_Math_and_Systems/1C.md)：先看多卡通信与显存共享，理解后面做训练调优时为什么会碰到通信边界。

## Task1-6 路线

| Task | 内容 | 章节 |
|:---|:---|:---|
| Task1 | PyTorch 与基础组件入口 | [00 PyTorch Warmup](../../02_PyTorch_Algorithms/00_PyTorch_Warmup.ipynb)、[01 RMSNorm](../../02_PyTorch_Algorithms/01_RMSNorm_Tutorial.ipynb)、[02 SwiGLU](../../02_PyTorch_Algorithms/02_SwiGLU_Activation.ipynb) |
| Task2 | 位置编码与注意力 | [03 RoPE](../../02_PyTorch_Algorithms/03_RoPE_Tutorial.ipynb)、[04 Attention MHA/GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb) |
| Task3 | SFT 数据与损失口径 | [01 SFT Data and Loss](./01_sft_data_and_loss.md) |
| Task4 | LoRA / PEFT 设计 | [02 LoRA / PEFT Design](./02_lora_peft_design.md) |
| Task5 | 训练控制与端到端实验 | [03 Training Control](./03_training_control.md)、[04 End-to-End Experiment](./04_end_to_end_experiment.md) |
| Task6 | 项目交付与图册 | [05 Project Delivery and Decision](./05_project_delivery_decision.md)、[06 Visual Assets](./06_visual_assets.md)，增强选读：[17 Autograd](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)、[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)、[26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `01` | SFT 数据三件套、response-only loss、shift logits | [01 SFT Data and Loss](./01_sft_data_and_loss.md) |
| `02` | LoRA 旁路、target modules、rank / alpha / dropout | [02 LoRA / PEFT Design](./02_lora_peft_design.md) |
| `03` | WSD + cosine decay 和 optimizer update 计数 | [03 Training Control](./03_training_control.md) |
| `04` | train / val batch、评估函数和最小训练报告 | [04 End-to-End Experiment](./04_end_to_end_experiment.md) |
| `05` | LoRA 项目配置、参数账本、指标汇总和采用建议 | [05 Project Delivery and Decision](./05_project_delivery_decision.md) |
| `06` | SFT 闭环图册、数据对齐、训练控制、项目决策 | [06 Visual Assets](./06_visual_assets.md) |

## 进阶占位

基础层只保留 6 页主结构。进阶层先占位，后续按需要再展开：

- `30` 长上下文微调
- `31` LoRA 变体理论
- `32` SFT 数据工程

## 推荐入口

- 如果你还没看结构组件，先按 Task1-3 过一遍，不需要深挖 MoE。
- 如果你只想学微调实践，直接从 `01 -> 02 -> 03 -> 04 -> 05 -> 06` 进入。
- 如果你已经跑通了基础闭环，再进入 `30 -> 31 -> 32` 看进阶 SFT。
- 如果你还想看 SFT 之后的偏好优化，接着去 [后训练与对齐专题](../post_training_alignment/intro.md)。
- 如果你关心小显存微调，在 `02` 后补 `26`，再回到 `05` 做项目收口。
- 如果你关心训练机制，在 `04` 后补 `17 -> 18`。

## 入口摘要

- 基础闭环路线：`01 -> 02 -> 03 -> 04 -> 05 -> 06`。
- 进阶占位：`30 -> 31 -> 32`。
- 增强选读：`17 -> 18 -> 26`，分别补梯度机制、loss backward 和 QLoRA。

## 正文页

- [casebook.md](./casebook.md)：按基础层 / 进阶占位展开“常见错误 / 排障清单 / 项目报告模板 / LoRA 配置选择”。
- [walkthrough.md](./walkthrough.md)：用一条 prompt/response 样本贯穿 `01-05`，最后落到 `06` 图册收口。

## 相关专题

- [大模型结构和原理专题](../model_architecture/intro.md)：当你需要先理解 LoRA 挂在哪些层上时先看这里。
- [后训练与对齐专题](../post_training_alignment/intro.md)：当你需要从 SFT 继续走到 DPO / GRPO / 偏好评测时先看这里。
- [量化与压缩专题](../quantization/intro.md)：当你需要把 QLoRA、PTQ / QAT 和部署压缩串起来时先看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当训练微调遇到 OOM、显存账本或 checkpointing 问题时看这里。
- [Profiling 专题](../profiling/intro.md)：当你需要证明 LoRA 是否真的更快、更省时看这里。

## 读法建议

- `09` 先看三件套对齐表，再写 TODO。
- `10` 先确认 target modules 和 trainable ratio，再看 merge。
- `11 / 12` 要一起理解：scheduler 按 optimizer update 计数，gradient accumulation 按 micro-batch 累积。
- `13` 是最小训练闭环，`60` 是项目交付闭环；不要把两者混成同一页。

## 专题状态

当前为专题入口页。`01-05 + 06` 的正文页和图册已完成第一轮增强，`30-32` 作为进阶占位保留，后续可继续补 casebook、walkthrough 和项目排障清单。
