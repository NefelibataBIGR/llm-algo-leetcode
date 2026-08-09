# 05 Block / Residual Path

## 页面目标

这一页解释现代 Transformer block 是怎么把 norm、attention、MLP 和 residual 组织成一条可训练、可扩展的信息流。

## 问题起点

单独看 `Norm`、`Attention`、`MLP` 都不够，因为真正决定模型行为的是它们如何被组装在一起。

block 组装关注的是：

- 各组件的先后顺序
- residual 如何让信息跨层传播
- pre-norm / post-norm 为什么会影响训练稳定性
- dense block 如何为 MoE 或结构技巧留接口

## 演化过程

### 早期 Transformer block

最早的 Transformer block 由 attention、FFN 和 residual 堆起来，核心目标是让序列建模可训练。

### Pre-Norm 成为主流

现代大模型更常见的是 pre-norm block：

- 先 norm，再做 attention / MLP
- 再通过 residual 把主干信息送回去

这样更利于深层训练稳定性。

### Dense block 的标准化

随着 LLaMA 类模型普及，很多现代 block 的组织方式逐渐收敛：

- RMSNorm
- self-attention
- residual
- RMSNorm
- MLP / SwiGLU
- residual

### 结构扩展

在这个基础上，还可以继续扩展：

- MoE 替换 MLP
- attention 的 head 结构变化
- 长上下文位置编码调整
- 真实实现里的局部 trick

## 代表模型

- `LLaMA`：现代 dense block 的参考样本
- `Gemma`：在 block 组织上强调工程可用性
- `DeepSeek`：在 block 之上继续叠加更激进的注意力或专家结构

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer block 的原始结构入口。 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 理解 pre-norm / post-norm 对 block 稳定性的影响。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 解释现代 block 中 norm 选择的趋势。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 block 组装的高质量样本。 |
| [NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456) | 代表在 block 内继续做归一化增强的思路。 |
| [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | 代表更深 block 的稳定化设计。 |

## 与 Part 02 的对应关系

- `01`、`02`、`03`、`04` 的组件都在这里重新组装
- `05` 是最直接的 block 级案例
- `06`、`07` 是把 MLP 换成 MoE 的扩展方式
- `08` 是真实模型里 block 变体的来源

## 可视化提示

建议画一张 block 总图，至少标出：

- input hidden state
- first norm
- attention
- residual add
- second norm
- MLP / SwiGLU
- second residual add

最好同时标出：

- pre-norm 的位置
- MoE 替换 MLP 的位置
- 真实实现中的局部变体

## 阅读建议

如果你已经看过：

- `norm_evolution.md`
- `attention_evolution.md`
- `mlp_ffn_evolution.md`
- `rope_position_encoding.md`

那么这一页就是把它们重新合成一张 block 图的地方。
