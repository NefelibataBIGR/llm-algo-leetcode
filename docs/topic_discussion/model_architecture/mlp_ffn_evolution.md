# 07 MLP / FFN Evolution

## 页面目标

这一页解释 Transformer 里的 FFN / MLP 为什么从最朴素的两层线性结构，逐步演化到 GELU、SwiGLU 和更复杂的门控设计。

## 问题起点

Attention 负责 token 之间的信息交互，但每个 token 自己内部的非线性变换主要依赖 MLP / FFN。

所以 MLP / FFN 解决的是：

- token 表示如何在通道维度上重组
- 非线性表达能力如何增强
- 计算成本和表达能力如何平衡

## 演化过程

### 经典 FFN

最基础的 Transformer FFN 是两层线性变换中间加激活函数。

### GELU / ReLU 变体

不同激活函数影响收敛速度和表达能力，是 FFN 的第一层演化。

### GLU / SwiGLU

门控 MLP 引入 gate 路径，让模型可以更灵活地控制信息通过。

现代 LLM 中，SwiGLU 很常见，因为它在效果和成本之间通常更均衡。

### 更复杂的 MLP 结构

有些模型会在 FFN 上继续做局部修改，比如：

- 宽度和比例调整
- 权重共享
- 与 MoE 结合

## 代表模型

- `LLaMA`：SwiGLU 是标准结构的一部分
- `Gemma`：同样强调高效且稳定的 MLP 设计
- `DeepSeek`：会在更大结构里继续组织 FFN / MoE 的关系

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 原始 Transformer FFN 的入口。 |
| [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) | 解释 GELU 为什么成为常见激活函数。 |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 解释门控 FFN 为什么会进入现代 Transformer。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [SwiGLU](https://arxiv.org/abs/2002.05202) | 现代 LLM 中最常见的 MLP 变体之一。 |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 高质量展示 SwiGLU 如何进入现代 block。 |
| [Switch Transformers](https://arxiv.org/abs/2101.03961) | 代表把 FFN 扩展成 MoE 的重要方向。 |

## 与 Part 02 的对应关系

- `02` 直接讲 SwiGLU
- `05` 里可以看到 MLP / FFN 如何进入 block
- `06`、`07` 里可以看到 FFN 如何进一步升级为 MoE

## 可视化提示

建议画一张 FFN 演进图：

- FFN
- FFN + GELU
- GLU / SwiGLU
- MoE-FFN

并标出：

- gate / up / down 路径
- hidden size 扩张比例
- 与 attention 分支的职责边界

## 阅读建议

如果你要继续扩展，建议接着看：

- `block_residual_path.md`
- `representative_models.md`
- `cross_module_comparison.md`
