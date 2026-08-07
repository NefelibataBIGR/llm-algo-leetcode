# 反向传播与训练机制正文

## 页面目标

这页把 `00 / 17 / 18 / 12 / 19 / 42 / 74` 串成一份基础排障清单，重点是：

- 梯度怎么回去
- backward 里哪些张量要留
- 梯度累积怎么影响训练节奏
- checkpointing / offload 为什么会改变 backward 代价
- 如何用 profiling 验证这些改动

## 常见检查项

| 检查项 | 你要确认什么 | 常见问题 |
|:---|:---|:---|
| Autograd 基础 | `grad_fn`、`saved_tensors`、`backward()` 是否看得懂 | 只会调用 API，不知道梯度怎么流 |
| Attention backward | `Q / K / V / P` 的反向顺序是否清楚 | 只记住公式，忘了中间量和保存点 |
| Activation / Loss backward | 哪些张量参与回传、哪些是监督区间 | loss 口径和 label 对齐错误 |
| 梯度累积 | micro-batch 和 effective batch 是否统一 | backward 次数对了，step 口径错了 |
| Checkpointing | 是重算，不是搬运 | 只看显存变小，不看时间代价 |
| Offload | 是搬运，不是重算 | 把 offload 和 checkpointing 混成一类 |
| Profiling | backward 热点和收益是否可复现 | 只看感觉，不看 baseline / after |

## 常见失败模式

- forward 看起来正常，但 backward 的梯度位置不对。
- 梯度累积做了，但 `optimizer.step()` 口径没统一。
- checkpointing 省了显存，但没有记录时间损失。
- offload 触发了，但带宽开销没算进结论。
- profiling 只记录了总时长，没有拆分 backward 热点。

## 机制对照

| 机制 | 解决什么 | 代价是什么 | 适合什么时候用 |
|:---|:---|:---|:---|
| Autograd / Backward | 梯度从 loss 传回参数 | 需要保存中间状态 | 所有训练任务 |
| Gradient Accumulation | 显存不够时扩大 effective batch | 训练节奏更长 | micro-batch 受限 |
| Checkpointing | 用重算换显存 | backward 变慢 | 激活占显存高 |
| Offload | 把部分状态搬离 GPU | 传输开销 | 显存特别紧张 |
| Profiling | 证明收益和瓶颈 | 额外观测成本 | 优化前后验证 |

## 任务映射

| Task | 关注点 |
|:---|:---|
| Task1 | 最小 autograd 和 backward 直觉 |
| Task2 | attention / activation / loss 的反向机制 |
| Task3 | 梯度累积和 backward 调度 |
| Task4 | checkpointing / offload 的显存代价 |
| Task5 | backward 热点和性能验证 |
| Task6 | 与训练微调和显存优化的联动 |

## 相关跳转

- 想看完整路线，回到 [反向传播与训练机制专题入口](./intro.md)。
- 想看连续故事线，去 [反向传播与训练机制深入阅读](./walkthrough.md)。
- 想看训练微调闭环，去 [训练微调闭环专题](../fine_tuning_training/intro.md)。
- 想看显存调优，去 [显存优化与性能调优专题](../memory_performance_tuning/intro.md)。

## 小结

反向传播不是一块独立孤立的知识，它会直接影响训练调度、显存代价和 profiling 结论。
