# 推理优化专题

## 专题定位

本专题用于串起推理优化主线：先看请求链路为什么慢，再看 prefill、decode、KV cache、调度和量化分别改的是哪一段，最后把判断收回 `66 / 67 / 68-70` 的 benchmark 与部署结论。这里重点关注 `TTFT / TPOT / throughput`；如果问题先表现为预算装不下，应优先转到显存优化专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part00 / Part01 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 推理结构地基 | `04 -> 05` | [01 Request Path and Metrics](./01_request_path_and_metrics.md) |
| Task2 | Attention 与 prefill | `20 + P1:03/14/24` | [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md) |
| Task3 | 解码算法 | `21 -> 23 -> 35 -> 36` | [03 Decoding Strategies](./03_decoding_strategies.md) |
| Task4 | KV Cache 与服务内存 | `P1:11 -> 22 -> 24 -> 34 -> 37` | [04 KV Cache and Scheduling](./04_kv_cache_and_scheduling.md) |
| Task5 | 量化推理与部署 | `P1:21 -> 25 -> 40 -> 41 -> 67` | [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md) |
| Task6 | benchmark 与项目复盘 | `39 -> 66 -> 68 -> 69 -> 70` | [06 Benchmark and Decision](./06_benchmark_and_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“这几个小节之间怎么串”“为什么先看这条线”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [推理优化正文](./casebook.md)，想按完整故事线走一遍就进 [推理优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责定位慢在哪里，[显存优化专题](../memory_performance_tuning/intro.md) 负责看预算与吞吐取舍，[量化与压缩专题](../quantization/intro.md) 负责看低比特路线，[编译与图优化专题](../compiler_graph_optimization/intro.md) 负责看 backend、fusion 和 kernel schedule 差异。
