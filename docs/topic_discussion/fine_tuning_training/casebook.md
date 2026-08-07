# 训练微调闭环正文

## 页面目标

这页把 `09-13 + 60` 变成一份可执行检查清单，重点是：

- 数据和 loss 是否正确
- LoRA 配置是否合理
- 训练控制是否按预期工作
- 项目输出是否足够支撑决策

## 关键检查项

| 检查项 | 你要确认什么 | 常见问题 |
|:---|:---|:---|
| 数据三件套 | `input_ids / attention_mask / labels` 是否对齐 | response 没进 loss、padding 参与 loss |
| Loss 口径 | shift logits 和 supervised token 是否一致 | label 对齐错误、EOS 位置错误 |
| LoRA 配置 | target modules、`r / alpha / dropout` 是否合适 | 挂错层、训练参数过少或过多 |
| 训练控制 | scheduler、accumulation、optimizer step 是否一致 | 按 micro-batch 计数、未除 accum_steps |
| 实验报告 | train / val、显存、速度、样例是否齐全 | 只看 loss，不看生成质量 |
| 项目交付 | adapter、tokenizer、config、结论是否保存 | 复现实验缺少关键 artifact |

## 常见失败模式

- loss 在降，但生成结果没有变好。
- 验证集 loss 好看，实际样例开始变模板化。
- 训练正常结束，但 adapter 保存后加载不一致。
- scheduler 和梯度累积的 step 口径不统一。
- 数据看起来能跑，但样本里有空 response 或格式脏样本。

## LoRA 配置判断

| 问题 | 先看什么 | 典型判断 |
|:---|:---|:---|
| 想快速验证微调 | `10` 的 target modules 和参数比例 | 先覆盖主要线性层，再做收缩 |
| 显存紧张 | `26` / `60` | 需要时再引入 QLoRA 或更小 batch |
| 训练不稳定 | `11` / `12` | 检查 scheduler 计数和 effective batch |
| 项目要交付 | `60` | 先看数据可信度，再看 artifact 和结果 |

## 任务映射

| Task | 关注点 |
|:---|:---|
| Task1 | SFT 数据和 labels |
| Task2 | LoRA 机制和挂载位置 |
| Task3 | scheduler / accumulation / 训练控制 |
| Task4 | 端到端实验与评估 |
| Task5 | 项目页和输出决策 |
| Task6 | 与显存、量化、profiling 的联动 |

## 相关跳转

- 想看完整路线，回到 [训练微调闭环专题入口](./intro.md)。
- 想看连续故事线，去 [训练微调闭环深入阅读](./walkthrough.md)。
- 想看结构前置，去 [大模型结构和原理专题](../model_architecture/intro.md)。
- 想看量化分支，去 [量化与压缩专题](../quantization/intro.md)。

## 小结

微调不是“把数据喂给模型跑一下”，而是数据、loss、训练控制和项目交付四件事一起闭环。
