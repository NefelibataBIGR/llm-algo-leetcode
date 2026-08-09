# 大模型结构和原理正文

## 页面目标

这页把 `01-08` 重新整理成一张“结构地图”，重点不是复述每节内容，而是回答：

- 为什么这个主题值得单独讲
- 每个组件在 block 里负责什么
- 输入输出张量大致长什么样
- 哪些内容是 dense block 的基础，哪些是 MoE 或结构技巧的扩展

## 当前定位

这一页现在更适合当“总览型正文骨架”：

- 想看模块演进，请去单页：`02_tokenization_embedding.md`、`03_norm_evolution.md`、`04_attention_evolution.md`、`05_rope_position_encoding.md`
- 想看 block 结构、MoE 扩展和 `01-08` 的映射，继续留在这一页
- 想看代表模型如何串起来，再回到 `walkthrough.md`

## 叙事骨架

如果把这个专题写成一段连续叙事，主线应该是：

`为什么先看结构 -> hidden state 怎么穿过 block -> dense block 怎么组装 -> MoE 怎么替换 MLP -> 真实实现和教科书哪里不同`

这条线的价值在于，它把 `01-08` 从“分别解释组件”变成“共同解释一个 block”。

再往前推一步，这条线本身也在回答三个问题：

- 为什么需要单独专题：因为结构决定了训练、推理、显存和微调的共同边界。
- 为什么不是只列文件：因为文件只是入口，真正要组织的是“组件之间的因果关系”。
- 为什么要强调取舍：因为每个结构选择都会改变数值稳定性、上下文建模和计算代价。

## 文献锚点

| 文献 | 主题对应 | 读它的理由 |
|:---|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | `05` | 看一个现代 LLM block 是如何把 norm、attention、MLP 组织起来的。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | `01` | 看为什么 RMSNorm 在大模型里是稳定且高效的选择。 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | `03` | 看 RoPE 如何把位置信息注入到 attention 里。 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | `02` | 看门控 MLP 为什么比普通 FFN 更适合现代 Transformer。 |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | `06 / 07` | 看 MoE 为什么要显式处理 router 和 load balancing。 |

## 可视化提示

这一页最值得画的图，是一张“LLaMA Block 拆解图”。

建议至少包括：

- 输入 hidden state
- 第一层 RMSNorm
- Attention 分支
- Residual 回路
- 第二层 RMSNorm
- SwiGLU / MLP 分支
- 可选 MoE 替换位置
- 真实实现中的 norm / residual 变体提示

这张图的目标不是复刻论文原图，而是让学习者能把 `01-08` 的每一节放到同一张图里。
如果后续要继续加厚，优先补的是 hidden state 的流向标注、Q/K/V 的位置标注、以及 MoE 替换 dense MLP 的分界线。

## 结构速记

| 组件 | 主要职责 | 你应该盯住什么 |
|:---|:---|:---|
| RMSNorm | 稳定激活尺度 | 归一化方向、残差前后位置 |
| SwiGLU | 提供门控 MLP | gate / up / down 三条分支 |
| RoPE | 给注意力注入位置信息 | Q / K 如何旋转、在哪一层生效 |
| Attention | 做 token 间交互 | MHA / GQA 的 head 关系、QKV 形状 |
| LLaMA Block | 组装基础结构 | Pre-Norm、Attention、MLP、Residual |
| MoE | 用专家替换 dense MLP | Router、Top-K、负载均衡 |
| Architecture Tricks | 适配真实模型实现 | weight tying、局部变体、命名差异 |

## 常见误区

- 只记住组件名字，不知道它在 block 的哪一段。
- 只看 Attention，不看 residual 和 pre-norm 的位置。
- 以为 MoE 只是“更大模型”，忽略 router 和负载均衡。
- 把结构技巧当成独立模块，而不是真实实现中的局部修正。

## 章节映射

| 章节 | 结构问题 | 常见判断 |
|:---|:---|:---|
| `01` | norm 放在哪 | 先看输入输出尺度是否稳定 |
| `02` | gate 怎么分支 | 先看 up / gate / down 三路关系 |
| `03` | 位置编码放在哪 | 先看 Q / K 是否共同旋转 |
| `04` | attention 怎么接上下文 | 先看 head 关系和 QKV 形状 |
| `05` | block 怎么串起来 | 先看 norm、attention、MLP、residual 的顺序 |
| `06` | router 怎么分配 token | 先看 Top-K 和专家路由 |
| `07` | 为什么需要 balance loss | 先看负载是否偏斜 |
| `08` | 结构 trick 怎么影响实现 | 先看真实源码和命名差异 |

## 典型检查清单

- 你能不能从 `05` 画出一条完整 dense block 数据流。
- 你能不能说清楚 `04` 里 Q / K / V 的 head 关系。
- 你能不能解释 `03` 的 RoPE 为什么作用在 Q / K 上。
- 你能不能区分 `06 / 07 / 09` 的 router、balance loss 和 sparse routing 各自解决什么问题。
- 你能不能把 `08` 的技巧放回具体模型实现里，而不是孤立记忆。

## Task 映射

| Task | 关注点 |
|:---|:---|
| Task1 | 基础归一化和激活 |
| Task2 | 位置编码和注意力 |
| Task3 | block 组装 |
| Task4 | MoE 扩展 |
| Task5 | 真实模型结构技巧 |
| Task6 | 与训练 / 推理 / 显存专题的衔接 |

## 相关跳转

- 想看完整路线，回到 [大模型结构和原理专题入口](./intro.md)。
- 想看连续故事线，去 [大模型结构和原理深入阅读](./walkthrough.md)。
- 想看训练闭环，去 [训练微调闭环专题](../fine_tuning_training/intro.md)。
- 想看推理侧影响，去 [推理优化专题](../inference_optimization/intro.md)。

## 样板说明

这一页作为样板正文，后续其他横向专题可以直接参考它的三层结构：

1. `叙事骨架`
2. `文献锚点`
3. `可视化提示`

如果某个专题暂时没有足够多文献，也至少要先把叙事骨架和可视化提示补出来。文件列表只能说明“有哪些入口”，不能说明“这些入口之间是什么关系”。

## 小结

这部分的目标不是背公式，而是把 `01-08` 变成一张可以拿来解释 block、attention 和 MoE 的结构图。
