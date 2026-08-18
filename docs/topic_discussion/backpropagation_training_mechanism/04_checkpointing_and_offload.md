# 04. Checkpointing 与 Offload | Checkpointing and Offload

## 页面目标

这一页讲两种最重要的显存优化策略：重算换显存和搬运换显存。

## 核心问题

### 1. checkpointing 做了什么

它不保存所有激活，而是只保存少量检查点。反向传播需要时，再从检查点重新计算中间段。

### 2. offload 做了什么

它把部分状态从 GPU 搬到别的存储层，比如 CPU 或更慢的内存层。

### 3. 它们有什么区别

checkpointing 是重算，offload 是搬运。两者都在省 GPU 显存，但代价来源不同。

## 机制分解

checkpointing / offload 不是同一个维度的方案：

- checkpointing 改变的是“前向状态要不要保留”
- offload 改变的是“状态保留在哪里”
- 两者都能省 GPU 显存，但一个吃算力，一个吃带宽

所以它们的边界一定要先看清：

- 如果显存不够但算力还富余，checkpointing 往往更直接
- 如果显存特别紧但重算已经太贵，offload 才更有价值
- 如果带宽太弱，offload 可能把瓶颈从显存换成传输

![Checkpointing 取舍图](/topic_discussion/backpropagation_training_mechanism/checkpointing_tradeoff.svg)

![Offload 取舍图](/topic_discussion/backpropagation_training_mechanism/offload_tradeoff.svg)

## 典型误区

- checkpointing 省的是激活显存，不是参数显存和优化器状态显存。
- offload 不是 checkpointing 的另一种说法。
- 是否开启这两类方案，不能只看显存，还要看 wall time 和带宽代价。

## 对应来源

- `19 Activation Checkpointing and Activation Offload`
- `42 Activation Offload`

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | checkpointing 的经典起点。 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | offload / partition / memory hierarchy 的系统起点。 |
| [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840) | 看 CPU offload 如何被纳入训练系统。 |
| [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857) | 看 GPU / CPU / NVMe 分层如何继续扩展 offload 路线。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint) | 看哪些张量可以通过重算从账本里拿掉。 |

## 阅读建议

- 先把 checkpointing 和 offload 区分开。
- 这页的重点是代价模型，不是 API 语法。
