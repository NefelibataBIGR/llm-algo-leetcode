# 06. Visual Assets | Visual Assets

## 页面目标

这一页把 SFT 闭环里最关键的图收起来，作为专题收口页。

## 图册

### 1. SFT 闭环总图

![SFT 闭环总图](/topic_discussion/fine_tuning_training/sft_flow_overview.svg)

这一张图回答的是：从 data trio 到 LoRA，到 training control，到 eval，再到 project decision，闭环是怎么串起来的。

### 2. 数据对齐图

![数据对齐图](/topic_discussion/fine_tuning_training/data_alignment.svg)

这一张图回答的是：prompt、response、labels、mask 和 response-only loss 的边界在哪里。

### 3. LoRA 与训练控制图

![LoRA 与训练控制图](/topic_discussion/fine_tuning_training/lora_training_control.svg)

这一张图回答的是：adapter、scheduler、accumulation 和 optimizer step 怎么互相配合。

### 4. 项目决策图

![项目决策图](/topic_discussion/fine_tuning_training/project_decision.svg)

这一张图回答的是：什么样的实验结果足够支撑项目决策。

## 你该怎么用这页

- 先看总图，确认闭环顺序。
- 再看数据对齐图，确认 loss 口径。
- 再看训练控制图，确认调度不打架。
- 最后看项目决策图，确认实验能不能交付。

## 相关跳转

- 回到 [监督微调（SFT）闭环专题入口](./intro.md)
- 回到 [监督微调（SFT）闭环正文](./casebook.md)
- 回到 [监督微调（SFT）闭环深入阅读](./walkthrough.md)
