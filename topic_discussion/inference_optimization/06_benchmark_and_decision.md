# 06. Benchmark and Decision | 端到端对比与选型

## 页面目标

这一页负责把前面的机制判断收束到 `66` 的 benchmark report 里。

## 问题起点

如果前面的 `01-05` 负责解释“慢在哪里、为什么会慢、有哪些候选动作”，那么 `06` 负责回答最后一个问题：**这次优化值不值得留下来**。

没有这一页，专题就会停在“知道很多技巧”；有了这一页，才算把技巧变成可复用的工程判断。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个变量。
- TTFT、TPOT、throughput 和 peak memory 是否一起报。

## 项目闭环

```text
workload config
      │
      ▼
prefill/decode metrics
      │
      ▼
bottleneck diagnosis
      │
      ▼
baseline vs candidate comparison
      │
      ▼
keep / tune / switch
```

## 为什么 `66` 是项目收口

`66` 的价值不在于再讲一遍机制，而在于把前面的判断塞回同一个 workload。只有在同一个模型、backend、batch、prompt tokens、generated tokens、dtype 和 cache policy 下，下面这些结论才有意义：

- FlashAttention 值不值得保留；
- speculative decoding 是真的更快，还是 acceptance 太低；
- KV cache 管理是否真的让并发收益上来；
- 量化到底是帮了忙，还是只是把代价换了个位置。

## 判定原则

- `keep`：收益不明显，或者代价太高。
- `tune`：方向对，但需要继续调参、调度或压缩。
- `switch`：收益稳定，且和目标场景匹配。

## 报告应该怎么写

一个合格的推理优化报告，至少要同时说明：

- 你改的是 prefill、decode、cache 还是量化；
- 这次改动对应的是哪一种瓶颈诊断；
- 指标变化是否和目标场景一致；
- 候选方案有没有引入新的副作用；
- 下一步是继续调参，还是保留当前 baseline。

![Benchmark decision flow](/topic_discussion/inference_optimization/benchmark_decision.svg)

## 报告清单

- workload 是否固定：模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy。
- 是否拆分 prefill 和 decode。
- 是否同时报告 TTFT、TPOT、throughput 和 peak memory。
- candidate 是否只改一个变量。
- 主要瓶颈是否能解释下一步动作。

## 文献与工程入口

- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
- Profiling 专题：当报告还无法证明慢点在哪里时先回去补 profiling。
- 推理优化 `01-05`：当报告还不能解释“为什么该切换/保留”时，回到对应问题页。

## 经典阅读入口

- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)

## 项目结论

`06` 不是新增机制页，而是把前面的判断变成最终结论。
