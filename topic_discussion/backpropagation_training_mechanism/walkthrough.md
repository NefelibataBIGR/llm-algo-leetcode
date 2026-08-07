# 反向传播与训练机制深入阅读

## 主故事线

一条完整的 backward 故事，可以按下面的顺序读：

`input -> forward -> save tensors -> loss -> backward -> gradient accumulation -> checkpointing / offload -> profiling`

这条线的重点不是背公式，而是把梯度怎么回去、为什么要保留某些状态、为什么 backward 会变慢讲清楚。

## 1. 从最小 autograd 开始

先看 `00`，建立最小直觉：

- forward 之后为什么会有 `grad_fn`
- 为什么 `loss.backward()` 会沿链路回传
- 自定义 `autograd.Function` 在做什么

## 2. 看 attention 的反向

进入 `17` 后，重点不是再记一次 attention 公式，而是：

- `Q / K / V / P` 的梯度从哪来
- 哪些中间量会被保存
- `gradcheck` 为什么有用

## 3. 看激活和损失的反向

进入 `18` 后，重点确认：

- activation backward 怎么影响梯度流
- loss backward 如何对应 supervision 区间
- `labels` / `ignore_index` / `reduction` 为什么会影响训练结论

## 4. 把 backward 放进训练循环

进入 `12` 后，要统一三件事：

- micro-batch 怎么累积成 effective batch
- backward 做几次，step 做几次
- loss / scheduler / optimizer 的计数口径是否一致

## 5. 看 backward 的显存代价

进入 `19` 后，重点是 checkpointing：

- 保存哪些张量
- 哪些张量可以重算
- 为什么 checkpointing 是“重算换显存”

再进入 `42`，理解 offload：

- 哪些状态搬到 CPU / 其他层
- 为什么它不是重算
- 为什么它会引入带宽代价

## 6. 用 profiling 收口

进入 `74` 后，不要只看总耗时，至少要确认：

- backward 热点在哪
- checkpointing / offload 是否真的值得
- 梯度累积是否改变了训练节奏
- 优化前后是否有可复现收益

## 典型分支

### 分支 1：梯度不对

先回到 `00 -> 17 -> 18`，不要先改训练调度。

### 分支 2：batch 受限

先看 `12`，确认是不是需要梯度累积，而不是直接改模型。

### 分支 3：显存不够

先看 `19 -> 42`，区分是重算问题还是搬运问题。

### 分支 4：性能退化

先看 `74`，确认退化出现在 forward、backward 还是通信。

## 阅读建议

1. 先从 `00` 看最小 backward。
2. 再看 `17 / 18`，把反向链路打实。
3. 再看 `12`，把 backward 放进训练节奏。
4. 再看 `19 / 42`，把 backward 和显存策略连起来。
5. 最后看 `74`，把这些机制放回 profiling 闭环。

## 和其他专题的连接

- 接到 [训练微调闭环专题](../fine_tuning_training/intro.md) 时，重点是 backward 如何影响 loss、scheduler 和项目闭环。
- 接到 [显存优化与性能调优专题](../memory_performance_tuning/intro.md) 时，重点是 checkpointing / offload / accumulation 的资源代价。
- 接到 [Profiling 专题](../profiling/intro.md) 时，重点是 backward 热点和收益验证。
