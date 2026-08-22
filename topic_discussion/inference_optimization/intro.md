# 推理优化专题

## 专题定位与 Infra 层定位

本专题串起推理优化主线：先定位请求链路中的 prefill、decode、KV Cache 和调度瓶颈，再比较解码策略、缓存策略、量化和 Serving 配置，最后回到 `66 / 67 / 68-70` 的 benchmark，形成可验证的部署决策。核心指标包括 `TTFT`、`TPOT`、吞吐、并发、P99、峰值显存和任务质量。

从 LLM Infra 五层看，推理优化的主要落点是：

- **Infra-L3 框架与运行时**：请求执行、prefill/decode 调度、KV Cache 管理和批处理；
- **Infra-L4 服务与模型优化**：推理引擎、Serving、量化部署、缓存策略和服务端指标；
- **Infra-L2 系统软件与加速库**：Attention kernel、低比特 kernel、编译和算子实现；
- **Infra-L1 硬件与基础设施**：GPU 算力、显存容量、带宽和互联拓扑。

因此，推理优化不是单独修改某一层：解码策略主要改变请求与模型执行行为，KV Cache 和调度主要影响 Infra-L3/Infra-L4，量化通常跨越 Infra-L2–Infra-L4，Attention/kernel 则需要结合 Infra-L2 实现和 Infra-L4 端到端服务结果判断。Infra-L5 的模型版本管理、灰度发布、扩缩容和流量治理属于平台交付，不是本专题的核心实验范围。

如果问题首先表现为“请求如何更快完成”，进入本专题；如果首先表现为“显存预算装不下”，转到显存优化专题；如果已经定位到单个 kernel 或图融合瓶颈，再进入编译与图优化专题。

## 同一内容的推理目标

本专题把解码、KV Cache、调度和量化看成“请求执行策略”：核心问题是一个模型如何在固定 workload 下更快、更稳地完成请求。量化在这里首先是部署策略，需要同时比较 backend、dtype、TTFT、TPOT、吞吐、P99、显存和任务质量；显存下降不能直接推出服务变快。详细的机制对照和指标判断见[推理优化正文](./casebook.md)。

与显存专题共享同一来源 Notebook 时，所有学习者先理解共同机制，再按目标选择指标：推理目标关注请求链路、服务并发和端到端体验，最终输出是 backend / 策略 / dtype 的服务选型结论。

## 推荐入口

推荐从 Part 02 的 [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) 开始；需要硬件和 Attention 前置时回补 Part 01 的 GPU、显存与访存内容。

## 前置阅读

- Part 01：GPU 架构、显存访问和 KV Cache 基础。
- Part 02：优先完成 2.6，再进入 2.7-2.8 的 serving、量化和调度内容。
- 运行真实 backend 前，先阅读 [使用指南](../../docs/guide.md) 和具体 Notebook 的环境说明。

## 主学习线

`Task0-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 / 项目入口 | 专题正文 |
|:---|:---|:---|:---|
| Task0 | 推理结构与请求链路基础 | [04 Attention / MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)；补充请求链路、TTFT、TPOT、throughput | [01 Request Path and Metrics](./01_request_path_and_metrics.md) |
| Task1 | GPU 硬件与推理性能基础 | [Part 01:03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)；提前了解 [Part 01:11 KV Cache Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) | [01 Request Path and Metrics](./01_request_path_and_metrics.md) |
| Task2 | Attention 访存瓶颈与 Prefill | [Part 01:14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) → [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)；扩展 [Part 01:24 SRAM Optimization](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)、[34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) | [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md) |
| Task3 | Decode 与生成策略 | [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)；扩展 [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)、[35 Multi-Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)、[36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)；扩展项目 [68 Speculative Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb) | [03 Decoding Strategies](./03_decoding_strategies.md) |
| Task4 | KV Cache 与推理服务内存管理 | [Part 01:11 KV Cache Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) → [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)；扩展 [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)、[38 PD Disaggregation](../../02_PyTorch_Algorithms/38_Prefill_Decode_Disaggregation.ipynb)；核心项目 [69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)；扩展项目 [70 Serving Scheduler](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) | [04 KV Cache and Scheduling](./04_kv_cache_and_scheduling.md) |
| Task5 | 量化推理与部署 | [Part 01:21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)；核心项目 [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)；扩展 [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md) |
| Task6 | 综合 benchmark 与项目决策 | 核心项目 [66 Inference Performance](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)；可选扩展 [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[68 Speculative Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)、[70 Serving Scheduler](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) | [06 Benchmark and Decision](./06_benchmark_and_decision.md) |

## 正文与跳转

先按上面的 `Task0-6` 走 notebook 主线；其中 `Task0-2` 主要建立结构、硬件和 Prefill 基础，`Task3-5` 进入 Decode、Cache、Serving 和量化策略，`Task6` 再用 66 做综合项目收口。带有“扩展”标记的内容不是主线前置，可按目标选读。遇到“这几个小节之间怎么串”“为什么先看这条线”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [推理优化正文](./casebook.md)，想按完整故事线走一遍就进 [推理优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责定位慢在哪里，[显存优化专题](../memory_performance_tuning/intro.md) 负责看预算与吞吐取舍，[量化与压缩专题](../quantization/intro.md) 负责看低比特路线，[编译与图优化专题](../compiler_graph_optimization/intro.md) 负责看 backend、fusion 和 kernel schedule 差异。

## 项目结论

项目按“主题验证 → 综合决策”分层：

- **核心综合项目：** [66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)，统一比较 backend、workload、TTFT、TPOT、吞吐、P99 和峰值显存。
- **主题项目：** [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)。它们分别属于 Task5 和 Task4，先验证单一策略，再由 66 做综合比较。
- **扩展项目：** [68 推测解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)。它们适合在具备真实 backend 或明确请求 workload 后选做。

没有 GPU 或真实 backend 时，可以先完成机制 Notebook 和 CPU-first 模板；接入 vLLM / SGLang 后，再将 66–70 升级为 Practice-P2 实验。最终统一使用 `accept / tune / reject` 输出策略判断。

## 环境与验证

基础机制可 CPU-first；真实吞吐、TTFT、TPOT 和 backend 对比需要 GPU 以及匹配的 vLLM / SGLang 运行环境。实验结论应同时记录模型、后端、数据类型、序列长度、并发度和结果文件。
