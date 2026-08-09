# 06. Benchmark and Trade-off Decision | Benchmark 与取舍决策

## 页面目标

这一页负责把前面的显存判断收束到收益验证：显存到底省了多少、时间赔了多少、最终值不值得留下来。

## 问题起点

如果没有验证页，显存专题很容易停在“某个技巧让峰值变小了”。但工程上真正有意义的问题是：

- 这次优化是否只是把显存从一个对象搬到另一个对象；
- 它是否把吞吐和延迟赔得过多；
- 它是否真的扩大了 batch、上下文或部署范围。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个变量。
- 峰值显存下降是否伴随吞吐、延迟或稳定性变化。

## 为什么 benchmark 是最后一页

显存优化通常横跨训练、推理和部署三条线。只有在同一 workload、同一指标口径下比较，下面这些结论才有意义：

- checkpointing 是否值得；
- offload 是否只是把问题换成传输等待；
- KV cache quantization 是否真的让上下文或并发收益上来；
- 量化或 paging 是否把时间成本赔过头。

## 判定原则

- `keep`：显存收益不明显，或者副作用过大。
- `tune`：方向对，但需要继续调 batch、调度、搬运或压缩参数。
- `switch`：显存收益稳定，且时间代价可接受。

## 报告应该怎么写

一个合格的显存优化报告至少要同时说明：

- 你动的是哪一个资源对象；
- 峰值显存变化了多少；
- 吞吐、延迟和稳定性怎么变；
- 这次变化有没有扩大 batch、上下文或部署空间；
- 最终是继续保留、继续调优，还是换方案。

![Memory benchmark decision flow](/topic_discussion/memory_performance_tuning/memory_benchmark_decision.svg)

## 文献与工程入口

- `73` Training Performance Analysis
- `74` Profiling Driven End-to-End Optimization
- `67` Quantized Inference and Deployment

## 典型阅读入口

- [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md)
- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md)

## 小结

显存优化只有在 benchmark 里仍然划算时，才算真正成立。
