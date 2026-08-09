# 05. 梯度累积、训练闭环与 Profiling | Gradient Accumulation, Training Loop and Profiling

## 页面目标

这一页把 backward 放回训练节奏里，回答为什么梯度累积、调度和 profiling 必须一起看。

## 核心问题

### 1. 为什么要梯度累积

因为大 batch 更稳定，但一次性塞进显存可能放不下。

### 2. 它怎么工作

把一个 batch 切成多个 micro-batch，多次 backward，最后只 step 一次。

### 3. 它改变了什么

它改变的是训练节奏和 effective batch，不是参数规模，也不是优化器原理。

## 机制拆解

gradient accumulation 里最容易混淆的是三个概念：

- `micro-batch`：每次真正送进 forward / backward 的最小批次
- `accumulation steps`：累积多少个 micro-batch 后再做一次参数更新
- `effective batch`：多个 micro-batch 累加之后，训练算法真正看到的总 batch 规模

所以，梯度累积不是“把 batch 变大”这么简单，而是把 `forward/backward` 的执行次数和 `optimizer.step()` 的执行频率拆开。

## 训练闭环

当你把前面的机制合起来看，训练闭环就变成：

- 先判断瓶颈是梯度、显存还是调度
- 再决定是积累、重算还是 offload
- 最后用 profiling 验证优化是否真的成立

![训练闭环决策图](/topic_discussion/backpropagation_training_mechanism/training_decision_flow.svg)

## 典型误区

- 梯度累积不会改变模型结构，它改变的是训练调度。
- `backward` 次数和 `step` 次数不是一回事，很多训练 bug 就出在这里。
- profiling 不是优化本身，它只是把瓶颈说清楚。

## 对应来源

- `12 Gradient Accumulation`
- `13 End-to-End Fine-Tuning Experiment`
- `14 RLHF PPO Memory`
- `74 Profiling-Driven End-to-End Optimization`

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) | 帮助理解 batch / step / generalization 为什么会纠缠在一起。 |
| [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://arxiv.org/abs/1705.08741) | 说明训练步数、更新频率和大 batch 训练之间的关系。 |
| [Enabling Large Batch Size Training for DNN Models Beyond the Memory Limit While Maintaining Performance](https://arxiv.org/abs/2110.12484) | 直接对应 micro-batch / gradient accumulation 的工程背景。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.profiler](https://docs.pytorch.org/docs/main/profiler) | 训练闭环里最直接的验证工具，帮助把“感觉慢”变成“哪里慢”。 |

## 阅读建议

- 先把 accumulation 和 step 区分开。
- 再把训练闭环放回 profiling。
- 如果你已经知道 batch / update cadence 的关系，就重点看验证方法。
