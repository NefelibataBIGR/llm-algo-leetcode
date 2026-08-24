# 05. Project Delivery and Decision | 项目交付与决策

## 页面目标

这一页回答的是：SFT 微调完成后，哪些产物必须保留，什么样的结果才值得交付和采用。

## 你要先确认什么

- adapter 是否可保存、可复现。
- tokenizer 和 config 是否和训练时一致。
- 项目报告是否说明了数据、训练和评估口径。
- 结果是否足以支撑采用决策。

## 演化路径

项目交付不只是“训完了”，而是要把实验结果整理成可复现的资产。

1. 保存 adapter 和必要配置。
2. 保存 tokenizer 和数据处理口径。
3. 保存 train / val 结果和代表性样例。
4. 写清楚采用或不采用的理由。
5. 如果后续要扩展，再把结论接回进阶占位。

这一页是 SFT 闭环的最后一环。
没有它，前面训练得再完整，也很难形成真正的项目结论。

## 常见误区

- 只保存权重，不保存 tokenizer 和 config。
- 报告里没有说明数据和 loss 口径。
- 只给出 loss 曲线，没有样例和结论。
- 结果能复现，但无法解释为什么值得采用。

## 经典阅读入口

- [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [26 QLoRA and 4bit Quantization](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)
- [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)

## 前置关系

- 先看 `04`，确认实验已经闭环。
- 再看 `05`，把实验变成交付决策。

## 项目结论

项目交付不是训练的附属步骤，而是闭环的一部分。
只有能交付、能复现、能解释的结果，才算真正完成 SFT。
