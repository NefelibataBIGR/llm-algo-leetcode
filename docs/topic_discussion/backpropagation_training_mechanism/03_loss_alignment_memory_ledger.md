# 03. Loss Backward、标签对齐与显存账本 | Loss Backward, Label Alignment and Memory Ledger

## 页面目标

这一页把训练里最容易出错的两件事放在一起看：监督口径到底在哪里，和 backward 为什么会把显存吃满。

## 核心问题

### 1. 为什么 `mask / shift / ignore_index` 很重要

训练里不是所有 token 都应该参与监督。prompt、padding、response、EOS 需要不同口径，否则 loss 会对错位置。

### 2. next-token loss 怎么对齐

自回归训练里，当前位置的 logits 预测下一个 token，所以必须做 shift。

### 3. 为什么激活会占住显存

只要某个中间量在 backward 还要用，它就不能随便丢。激活、参数、梯度、优化器状态和系统缓冲一起构成训练显存账本。

## 机制分解

label alignment 里最关键的不是“有没有算 loss”，而是“loss 在哪里算”：

- prompt 通常不应该被当成监督目标
- response 区间才是主要学习对象
- padding 必须被排除，否则梯度会被无意义 token 污染
- causal LM 里还要做 shift，确保当前位置预测下一个 token

显存账本里最容易漏掉的是一个事实：训练显存不只有 activation。

- 参数
- 梯度
- optimizer state
- 临时 workspace
- 通信缓冲区

都会参与训练预算。

## 典型误区

- `ignore_index` 不是“随便填个值”，它是监督边界的一部分。
- shift 不只是 shape 对齐，它决定预测目标到底是谁。
- 看到显存满了，不代表全部都是 activation。

## 对应来源

- `09 SFT Training Loop`
- `18 Activation and Loss Backward`
- `19 Activation Checkpointing and Activation Offload`

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 反向信号如何塑造表示，是理解 loss 监督边界的基础语境。 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | decoder self-attention 和 causal mask 是理解自回归监督口径的起点。 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 先理解显存账本为什么会逼出 checkpointing。 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 看参数、梯度和 optimizer state 如何一起进入显存预算。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | 直接看 `ignore_index`、reduction 和 target 口径。 |

## 阅读建议

- 先确认监督区间，再谈模型是否学得好。
- 如果你已经知道 next-token prediction，就重点看 mask / shift / ignore_index。
