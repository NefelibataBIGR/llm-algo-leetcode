# 显存优化与性能调优正文

这页只做显存问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 判断表

先分清问题在训练侧还是推理侧，再分清主要资源对象是 `activation`、`optimizer state`、`KV cache` 还是临时 buffer，最后判断省下来的显存有没有把时间代价一起控制住。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 训练前几步正常，中后段突然 OOM | `training activation pressure` | [02](./02_training_memory_pressure.md), [03](./03_checkpointing_and_offload.md) | 检查 batch、accumulation、checkpointing、offload |
| 推理能跑，但 cache 一直涨，batch 上不去 | `inference cache pressure` | [04](./04_inference_cache_and_memory_budget.md) | 检查 paging、prefix reuse、eviction、KV cache quant |
| 峰值显存下降了，但 benchmark 没改善 | `trade-off mismatch` | [06](./06_benchmark_and_tradeoff_decision.md) | 比较 peak memory、step time、TTFT、TPOT、throughput |
| 理论账本和实测差很多 | `ledger mismatch` | [01](./01_vram_ledger_and_metrics.md), [06](./06_benchmark_and_tradeoff_decision.md) | 对齐理论账本、运行时 buffer、碎片和流程开销 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `activation` | 训练侧主峰值是不是来自前反向中间状态 | 把所有问题都归到 batch 太大 |
| `optimizer state` | 更新状态是不是把预算继续抬高 | 只看参数量，不看更新状态驻留 |
| `KV cache` | 推理侧预算是不是被缓存增长顶高 | 看到延迟差就直接改 decode |
| `peak memory + time` | 省显存是否把时间和吞吐一起赔掉 | 峰值降了就默认 adopt |

最终判断不该停在“省了多少显存”，而要落回“系统是不是因此更可运行、更稳定、更值得保留”。

## 小结

这页的职责不是列出更多省显存的方法名，而是把显存问题里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`，项目证明留给 benchmark 和项目页。
