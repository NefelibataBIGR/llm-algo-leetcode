# 06 项目决策与交付

## 页面目标

这一页把 `02-05` 的方法、数据和评测重新收束成项目决策问题：  
什么时候值得 adopt，什么时候只是继续 tune，什么时候应该 reject。

## 问题起点

项目页真正需要的不是“我学了哪种方法”，而是：

- 当前数据形态适合哪条路线？
- 训练和系统代价是否值得？
- 评测是否真的支持结果更优？
- 最终项目交付该给出什么结论？

## 决策框架

可以按下面四步判断：

1. **先看前置是否满足**  
   SFT / LoRA 基线是否稳定，偏好数据是否可用。
2. **再看方法是否匹配**  
   是 pairwise preference，还是 group-wise candidate comparison。
3. **再看评测是否一致**  
   指标是否真的反映对齐收益。
4. **最后给出结论**  
   `adopt / tune / reject`，而不是只报 loss。

## 项目页对应

| 项目页 | 你要确认什么 |
|:---|:---|
| `84 DPO Preference Project` | preference pair、reference 口径、评测结果是否支持采用 DPO |
| `85 GRPO Groupwise Alignment Project` | 候选组构造、group-wise 结果和评测是否支持采用 GRPO |

## 可视化入口

![Post-Training Project Decision Board](/topic_discussion/post_training_alignment/project_decision_board.svg)

## 常见失败模式

- 方法选对了，但数据质量不足，结果不稳。
- 指标提升了，但无法解释是否真是偏好收益。
- 项目页有很多实验表，却没有明确 adopt / tune / reject 结论。

## 对应 Part02

- `84 DPO Preference Project`
- `85 GRPO Groupwise Alignment Project`
- 前置回跳：`15 / 16 / 50`

## 文献锚点

- DPO / GRPO 项目化实践资料。
- 对齐 benchmark 报告与项目采用标准说明。

## 小结

后训练专题真正的收口，不是“我知道几个方法”，而是“我能基于方法、数据和评测给出可交付的项目判断”。
