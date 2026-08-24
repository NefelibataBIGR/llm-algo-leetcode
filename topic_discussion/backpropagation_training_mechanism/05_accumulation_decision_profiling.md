# 05. 梯度累积、训练闭环与 Profiling | Gradient Accumulation, Training Loop and Profiling

## 页面目标

这一页把 backward 放回训练节奏里，回答为什么梯度累积、调度和 profiling 必须一起看。

本页的输出是训练闭环决策：明确 micro-batch、accumulation、optimizer step、AMP 和 profiling 的顺序，并把显存收益与吞吐、质量一起验证。

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

### AMP / BF16 / GradScaler / gradient clipping

真实训练里，梯度累积几乎不会单独出现，它通常和数值精度与稳定化策略绑在一起：

- `AMP / BF16`：改变的是前向、反向和参数更新所处的数值精度环境
- `GradScaler`：主要服务于 FP16 训练，避免小梯度在反向传播里直接下溢
- `gradient clipping`：负责在梯度已经算出来之后，限制极端更新步长

这几者和 accumulation 的关系要先分清：

- accumulation 改的是“多久 step 一次”
- mixed precision 改的是“这些张量用什么精度参与计算”
- GradScaler 改的是“loss 和梯度在反向阶段是否先做缩放”
- gradient clipping 改的是“step 之前是否限制梯度范数”

所以现代训练闭环更接近下面这条链：

`autocast forward -> scaled loss backward -> accumulation -> unscale(if needed) -> gradient clipping -> optimizer.step() -> profiler validate`

如果这一层没理顺，就很容易出现两类问题：

- loss 看起来正常，但梯度早就在低精度里下溢或爆掉
- accumulation 和 clipping / scaler 的调用时机错位，导致训练节奏和数值行为一起失真

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
- `BF16` 通常不依赖 `GradScaler`，`FP16` 才更依赖缩放保护。
- `gradient clipping` 应该发生在真正 `step` 之前，而不是每个 micro-batch 后都盲目执行。

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
| [torch.amp](https://docs.pytorch.org/docs/stable/amp.html) | 直接看 autocast、GradScaler 和 mixed precision 的官方接口。 |
| [clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | 看 gradient clipping 在 step 前应该如何调用。 |

## 阅读建议

- 先把 accumulation 和 step 区分开。
- 再把 `AMP / BF16 / GradScaler / clipping` 放回 step 之前的调用顺序里。
- 最后把训练闭环放回 profiling。
- 如果你已经知道 batch / update cadence 的关系，就重点看验证方法。

## 回到项目

将机制结论回填到 `73 Training Performance Analysis -> 76 Activation Checkpoint / Offload Benchmark`；如果需要预算与策略采用判断，再接入 `75 Memory Budget Compression Project`。基础机制页负责解释原因，项目页负责给出实测结论。
