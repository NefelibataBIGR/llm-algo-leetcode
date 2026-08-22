# 推理优化专题

## 专题定位

本专题用于串起推理优化主线：先看请求链路为什么慢，再看 prefill、decode、KV cache、调度和量化分别改的是哪一段，最后把判断收回 `66 / 67 / 68-70` 的 benchmark 与部署结论。这里重点关注 `TTFT / TPOT / throughput`；如果问题先表现为预算装不下，应优先转到显存优化专题。

## Infra 层定位

推理优化主要位于 `L3 框架与运行时`、`L4 服务与模型优化`，同时受 `L2 算子/编译` 和 `L1 硬件带宽与显存` 约束。解码策略主要改变模型与请求行为，Attention/kernel 主要改变 L2，KV Cache 与调度主要改变 L3/L4，量化和并行部署则跨越 L2-L5。

## 同一内容的推理目标

本专题把解码、KV Cache、调度和量化看成“请求执行策略”：核心问题是一个模型如何在固定 workload 下更快、更稳地完成请求。量化在这里是部署策略，需要同时比较 backend、dtype、TTFT、TPOT、吞吐、P99、显存和任务质量；显存下降不能直接推出服务变快。

与显存专题共享同一来源 Notebook 时，所有学习者先理解共同机制，再按目标选择指标：推理目标关注请求链路、服务并发和端到端体验，最终输出是 backend / 策略 / dtype 的服务选型结论。

## 推荐入口

推荐从 Part 02 的 [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) 开始；需要硬件和 Attention 前置时回补 Part 01 的 GPU、显存与访存内容。

## 前置阅读

- Part 01：GPU 架构、显存访问和 KV Cache 基础。
- Part 02：优先完成 2.6，再进入 2.7-2.8 的 serving、量化和调度内容。
- 运行真实 backend 前，先阅读 [使用指南](../../docs/guide.md) 和具体 Notebook 的环境说明。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 推理结构地基 | `04 -> 05` | [01 Request Path and Metrics](./01_request_path_and_metrics.md) |
| Task2 | Attention 与 prefill | `20 + Part 01:03/14/24` | [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md) |
| Task3 | 解码算法 | `21 -> 23 -> 35 -> 36` | [03 Decoding Strategies](./03_decoding_strategies.md) |
| Task4 | KV Cache、服务内存与请求 disaggregation | `Part 01:11 -> 22 -> 24 -> 34 -> 37 -> 38` | [04 KV Cache and Scheduling](./04_kv_cache_and_scheduling.md) |
| Task5 | 量化推理与部署 | `Part 01:21 -> 25 -> 40 -> 41 -> 67` | [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md) |
| Task6 | benchmark 与项目复盘 | `39 -> 66 -> 68 -> 69 -> 70` | [06 Benchmark and Decision](./06_benchmark_and_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“这几个小节之间怎么串”“为什么先看这条线”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [推理优化正文](./casebook.md)，想按完整故事线走一遍就进 [推理优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责定位慢在哪里，[显存优化专题](../memory_performance_tuning/intro.md) 负责看预算与吞吐取舍，[量化与压缩专题](../quantization/intro.md) 负责看低比特路线，[编译与图优化专题](../compiler_graph_optimization/intro.md) 负责看 backend、fusion 和 kernel schedule 差异。

## 项目结论

核心入口是 [66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)；扩展项目包括 [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[68 推测解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) 和 [70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)。

## 环境与验证

基础机制可 CPU-first；真实吞吐、TTFT、TPOT 和 backend 对比需要 GPU 以及匹配的 vLLM / SGLang 运行环境。实验结论应同时记录模型、后端、数据类型、序列长度、并发度和结果文件。
