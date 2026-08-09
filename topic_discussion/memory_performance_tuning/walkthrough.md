# 显存优化与性能调优深入阅读

## 主故事线

如果把这条线写成完整排障过程，它通常是这样：先是训练中后段突然 OOM，第一反应是缩 batch，但沿着 `12 -> 19 -> 42 -> 73` 一查，发现真正的问题是 effective batch、activation 和重算把显存压力持续抬高；接着换到推理场景，又发现 cache 随请求增长、延迟也在变差，于是顺着 `2.6 -> 22 -> 67` 看前缀复用、KV cache 和量化部署的权衡；然后回到 `06 -> 13 -> 73`，把理论账本和实测峰值对一遍，确认是不是还有隐藏开销；最后再用 `74 -> 67` 比较优化前后到底是“省了显存但赔了时间”，还是“资源和吞吐都更划算了”。

如果你已经知道问题落在哪一层，可以直接跳到对应编号页：

- [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md)
- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md)
- [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)

## 端到端案例

一个更完整的显存调优过程，通常是从“训练到中后段突然 OOM，推理 cache 也越跑越大”开始的。先沿着 `12 -> 19 -> 42 -> 73` 看训练侧 batch、activation、checkpointing 和 offload，确认是不是重算和中间状态把显存一点点抬上去；再沿着 `2.6 -> 22 -> 67` 看推理侧 KV cache、前缀复用和量化部署，确认是不是请求形态让显存增长更快；然后回到 `06 -> 13 -> 73`，把理论账本和实测峰值对一遍，确认是不是还有隐藏开销没有算进去；最后用 `74 -> 67` 比较优化前后的吞吐、延迟和显存占用，判断这次调整到底是“省了显存但赔了性能”，还是把系统拉到了更划算的平衡点。

## 阅读建议

- 先读长故事，再看正文里的资源对象、案例和清单。
- 如果你已经确定是训练 / 推理 / 验证中的某一类问题，可以直接看 [显存优化与性能调优正文](./casebook.md)。
- 如果你先要确认瓶颈，也可以先回到 [Profiling 专题](../profiling/intro.md)。
