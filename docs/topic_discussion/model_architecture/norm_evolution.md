# 02 Norm Evolution

## 页面目标

这一页解释归一化为什么会从 LayerNorm 一路演化到 RMSNorm、Pre-Norm 和更深层的稳定化设计。

## 问题起点

归一化解决的不是“让数值好看”这种表面问题，而是：

- 让深层网络更稳定
- 让 residual 路径更可控
- 让训练不那么容易发散

在大模型里，norm 的位置和类型直接影响：

- 梯度流动
- 收敛速度
- 训练稳定性
- block 级残差行为

## 演化过程

### LayerNorm 时代

LayerNorm 提供了最经典的归一化思路，先稳定激活再做后续计算。

### Pre-Norm / Post-Norm 讨论

Transformer 训练中，norm 放在 attention / MLP 前后，会直接影响梯度传播和深层训练稳定性。

### RMSNorm 时代

RMSNorm 去掉了均值中心化，只保留尺度归一化，常见于现代 LLM。

它的优势通常体现在：

- 计算更轻
- 在大模型里通常足够稳定
- 和 residual / block 设计搭配更自然

### 更深层的稳定化设计

- NormFormer 增加额外归一化来改善训练
- DeepNorm 试图让极深 Transformer 更稳定
- 一些模型还会结合 residual scaling 或局部 trick

## 代表模型

- `LLaMA`：RMSNorm 是其标准 block 组件之一
- `Gemma`：结构上也强调 norm 的稳定性和工程可用性
- `DeepSeek`：在现代大模型中继续沿用高效 norm 设计

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 归一化的基础入口，理解后续所有变体的起点。 |
| [Transformers without tears: Improving the normalization of self-attention](https://www.amazon.science/publications/transformers-without-tears-improving-the-normalization-of-self-attention) | 帮助理解 Transformer 里 norm 放置方式为什么会影响稳定性。 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 解释 RMSNorm 为什么成为现代 LLM 的常用选择。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456) | 展示额外归一化如何影响训练和性能。 |
| [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | 代表更深层网络的稳定化思路，适合看 norm 与深度的关系。 |

## 与 Part 02 的对应关系

- `01` 直接讲 RMSNorm
- `05` 里可以看到 norm 如何进入 block 组装
- `08` 里可以看到真实模型中 norm 的局部变体

## 可视化提示

建议画一张“norm 演进时间线”：

- LayerNorm
- Pre-Norm / Post-Norm
- RMSNorm
- NormFormer / DeepNorm

再补一张“block 内 norm 位置图”，标出：

- attention 前的 norm
- MLP 前的 norm
- residual 与 norm 的相对关系

## 阅读建议

如果你要继续扩展，建议接着看：

- `attention_evolution.md`
- `rope_position_encoding.md`
- `block_residual_path.md`
