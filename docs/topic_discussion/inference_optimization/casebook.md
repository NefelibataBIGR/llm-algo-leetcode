# 推理优化正文

这页只做推理问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 判断表

先分清问题在 `prefill`、`decode`、`cache / scheduling` 还是 `deployment`，再统一 `TTFT / TPOT / throughput / peak memory` 口径，最后回到同一 workload，把候选方案收成 `accept / tune / reject`。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 长 prompt 下首 token 明显变慢 | `prefill-bound` | [02](./02_prefill_and_attention_kernel.md) | FlashAttention、chunked prefill、prefix caching |
| 并发一高，生成速度掉下去 | `decode-bound` | [03](./03_decoding_strategies.md) | speculative decoding、multi-token decoding、decode scheduling |
| cache 一边跑一边涨，batch 上不去 | `memory-bound` | [04](./04_kv_cache_and_scheduling.md) | paging、prefix reuse、eviction、KV cache quant |
| 显存降了，但交互体验变差 | `deployment trade-off` | [05](./05_quantized_inference_and_deployment.md) + [06](./06_benchmark_and_decision.md) | 区分权重量化、KV cache quant、FP8，再回 benchmark |

显存已经接近预算时，优先把它当成硬约束处理；即使 decode 也慢，继续上 batch 或上下文都不可靠。

| 指标 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| `TTFT` | 首 token 是否被 prefill 拖慢 | 只看总时延，看不出 prefill 问题 |
| `TPOT` | decode 阶段每 token 是否过慢 | 把 decode 慢误判成模型整体慢 |
| `throughput` | 系统单位时间产出是否够高 | 只看吞吐，不看交互延迟 |
| `peak memory` | 当前配置是否还能继续推 batch / context | 不把它当硬约束，只看速度 |
| `prefill_share` / `decode_share` | 主要时间花在哪一段 | 没拆阶段，无法判断下一步该改哪里 |

`66` 的价值不在于再讲机制，而在于把这些指标放回同一 workload 比较 baseline 和 candidate。

## 小结

这页的职责不是列技巧，而是把症状、指标和下一步动作压成一张判断表。路线入口留给 `intro`，连续故事留给 `walkthrough`，项目证明留给 `66`。
