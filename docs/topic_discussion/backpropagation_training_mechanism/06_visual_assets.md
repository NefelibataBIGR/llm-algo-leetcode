# 06. Visual Assets | Visual Assets

## 页面目标

这一页把反向传播专题的关键图收口，方便你快速回看整条机制链路。

## 图册

### 1. 反向传播与训练闭环总图

![训练闭环决策图](/topic_discussion/backpropagation_training_mechanism/training_decision_flow.svg)

这张图负责把 backward、调度和 profiling 的关系拉直，适合作为专题总览。

### 2. Attention Backward 图

![Attention backward 图](/topic_discussion/backpropagation_training_mechanism/attention_backward.svg)

这张图负责把 `dV -> dP -> dS -> dQ / dK` 的反向顺序固定下来。

### 2.1 Naive vs FlashAttention Backward 对照图

![Naive vs FlashAttention Backward](/topic_discussion/backpropagation_training_mechanism/attention_backward_impl_compare.svg)

这张图负责说明现代实现差异：

- 哪些中间状态被完整保存
- 哪些状态在 backward 中重算
- 哪些路径通过 fused kernel 减少了中间写回

### 3. 显存账本图

![显存账本图](/topic_discussion/backpropagation_training_mechanism/activation_ledger.svg)

这张图负责说明 backward 为什么会先吃满显存，以及 activation / parameter / optimizer state 的差别。

### 4. Checkpointing 取舍图

![Checkpointing 取舍图](/topic_discussion/backpropagation_training_mechanism/checkpointing_tradeoff.svg)

这张图负责说明“重算换显存”的代价模型。

### 5. Offload 取舍图

![Offload 取舍图](/topic_discussion/backpropagation_training_mechanism/offload_tradeoff.svg)

这张图负责说明“搬运换显存”的代价模型。

### 6. 标签对齐图

![标签对齐图](/topic_discussion/backpropagation_training_mechanism/label_alignment.svg)

这张图负责说明 prompt、response、mask 和 loss 的监督边界。

## 使用方式

- 先看总图，确认机制链路。
- 再看 attention 图，确认梯度回传顺序。
- 再用对照图确认现代实现里哪些状态是“保存”、哪些是“重算”、哪些是“融合带过”。
- 再看显存账本图，确认为什么要做优化。
- 最后看两类取舍图和标签对齐图，确认怎么选方案。

## 相关跳转

- 回到 [反向传播与训练机制专题入口](./intro.md)
- 回到 [反向传播与训练机制正文](./casebook.md)
- 回到 [反向传播与训练机制深入阅读](./walkthrough.md)
