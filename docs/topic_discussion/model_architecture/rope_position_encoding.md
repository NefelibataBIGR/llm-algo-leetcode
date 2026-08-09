# 04 RoPE / Position Encoding

## 页面目标

这一页解释位置编码为什么从绝对位置走向相对位置，再走向 RoPE，以及为什么长上下文模型又必须继续改造 RoPE。

## 问题起点

attention 本身不带顺序感，因此位置编码是“让模型知道 token 顺序”的必要补充。

位置编码解决的问题包括：

- token 顺序如何进入 attention
- 相对位置信息如何表达
- 长上下文如何扩展而不显著退化

## 演化过程

### 绝对位置编码

最早的做法是给每个位置一个明确的编码，但它的可泛化性和长上下文适配都有限。

### 相对位置编码

相对位置方法更关注 token 之间的相对关系，便于更自然地表达上下文结构。

### RoPE

RoPE 把位置信息融入 query / key 的旋转关系中，成为现代 LLM 的常见默认选择。

它的优势通常体现在：

- 与 attention 结合自然
- 适合现代 LLM block
- 结构上便于扩展到更长上下文

### 长上下文扩展

当上下文长度上去之后，RoPE 通常需要进一步做缩放、插值或重参数化处理。

## 代表模型

- `LLaMA`：RoPE 是其标准位置编码选择之一
- `Qwen`：会结合长上下文和工程需求看位置编码策略
- `DeepSeek`：在更复杂的结构里继续使用或改造 RoPE 相关机制

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 位置编码的原始起点，理解绝对位置编码的基础。 |
| [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) | 相对位置编码的重要入口，适合和 RoPE 对照。 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE 的核心论文，是现代 LLM 位置编码的关键节点。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | 代表长上下文扩展中的插值思路。 |
| [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) | 代表 RoPE 伸缩与长上下文扩展的工程路线。 |
| [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) | 代表更激进的长上下文扩展方法。 |

## 与 Part 02 的对应关系

- `03` 直接讲 RoPE 的作用位置
- `05` 里可以看到 RoPE 如何嵌进 block
- `08` 里可以看到真实模型对 RoPE 的局部修正

## 可视化提示

建议画一张“位置编码演化图”：

- absolute position
- relative position
- RoPE
- RoPE scaling / interpolation / extension

最好同时标出：

- Q / K 上的位置变化
- 长上下文时为何需要重新标定

## 阅读建议

如果你要继续扩展，建议接着看：

- `block_residual_path.md`
- `representative_models.md`
- `cross_module_comparison.md`
