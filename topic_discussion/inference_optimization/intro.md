# 推理优化专题

## 页面导语

本专题围绕请求链路、KV Cache、解码、量化和 Serving，最终形成可验证的推理部署决策。

## 如何开始

推荐从 Part 02 的 [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) 开始；需要硬件和 Attention 前置时回补 Part 01 的 GPU、显存与访存内容。

- 必读前置：Part 01 的 GPU 架构、显存访问和 KV Cache 基础；Part 02 优先完成 2.6，再进入 2.7–2.8 的 serving、量化和调度内容。
- 按需回补：如果对 Attention、带宽或 KV Cache 不熟，先回补对应 Part 00 / Part 01 Notebook。
- 真实 backend 实验前：先阅读 [使用指南](../../docs/guide.md) 和具体 Notebook 的环境说明。

## 主学习线与分级

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

### 核心与扩展分级

核心路径先建立请求链路、访存和服务指标的共同口径，再完成一次可复查的单策略或单 backend 实验；扩展路径才进入真实服务能力、复杂 workload、多模型协作和跨策略比较。这样没有 GPU 或 backend 的学习者仍可完成主线，有 GPU 的学习者再逐步升级 Practice-P1 / Practice-P2。

| Task | 核心路径 | 扩展路径 | 环境级别 |
|:---|:---|:---|:---|
| Task0 | [04 Attention / MHA / GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)，理解请求中的 prefill、decode 和 token 生成 | 补充更复杂的 Attention 变体和请求链路拆分 | Practice-P0 |
| Task1 | [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) 与 [11 KV Cache Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) | [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb) 等硬件访存分析 | Practice-P0/P1 |
| Task2 | [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)，理解 prefill 访存瓶颈 | [24 SRAM Optimization](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)、[34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) | Practice-P0/P1 |
| Task3 | [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)，比较生成策略的基本代价 | [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)、[35 Multi-Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb)、[36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)、[68 Speculative Benchmark](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb) | 核心 P0/P1；扩展 P2 |
| Task4 | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) → [69 Prefix Cache](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb) | [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)、[70 Serving Scheduler](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) | 机制 P0/P1；真实 backend P2 |
| Task5 | [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) | [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) 和真实量化 backend | 核心 P0/P1；部署扩展 P2 |
| Task6 | [66 Inference Performance](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 完成固定 workload 的综合决策 | 用 [67](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[68](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[69](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)、[70](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb) 做真实 backend、并发 sweep 和多策略对照 | 核心 P1/P2；扩展 P2 |

核心路径的最低产物是固定 workload、baseline、候选指标和 `accept / tune / reject`；扩展路径还必须补充 backend 版本、真实服务启动方式、并发或请求分布 sweep，以及策略专属质量指标，例如 acceptance rate、cache hit rate 或 fairness。`66` 当前已完成真实 backend smoke test，但正式结论仍需要更大 workload 和重复运行。

## 学习方式与项目产出

先按上面的 `Task0-6` 走 Notebook 主线；核心路径用于建立机制和完成最小实验，扩展路径用于真实 GPU、backend、并发或复杂 workload。需要连续理解概念时阅读专题正文 `01-06`，需要判断表和项目分流时阅读[推理优化正文](./casebook.md)，需要完整串联路线时阅读[推理优化深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[Profiling 专题](../profiling/intro.md) 负责定位慢在哪里，[显存优化专题](../memory_performance_tuning/intro.md) 负责看预算与吞吐取舍，[量化与压缩专题](../quantization/intro.md) 负责看低比特路线，[编译与图优化专题](../compiler_graph_optimization/intro.md) 负责看 backend、fusion 和 kernel schedule 差异。

### 项目产出

项目按“主题验证 → 综合决策”分层：

- **核心综合项目：** [66 推理性能对比](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)，统一比较 backend、workload、TTFT、TPOT、吞吐、P99 和峰值显存。
- **主题项目：** [67 量化部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[69 前缀缓存](../../02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)。它们分别属于 Task5 和 Task4，先验证单一策略，再由 66 做综合比较。
- **扩展项目：** [68 推测解码](../../02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)、[70 服务调度](../../02_PyTorch_Algorithms/70_Serving_Scheduler_Benchmark.ipynb)。它们适合在具备真实 backend 或明确请求 workload 后选做。

没有 GPU 或真实 backend 时，可以先完成机制 Notebook 和 CPU-first 模板；接入 vLLM / SGLang 后，再将 66–70 升级为 Practice-P2 实验。最终统一使用 `accept / tune / reject` 输出策略判断。

## 环境与验证

基础机制可 CPU-first；真实吞吐、TTFT、TPOT 和 backend 对比需要 GPU 以及匹配的 vLLM / SGLang 运行环境。实验结论应同时记录模型、后端、数据类型、序列长度、并发度和结果文件。

开始真实实验前，先看[使用指南](../../docs/guide.md)中的环境边界；需要逐条执行时，使用[66–70 推理项目验证清单](../../docs/verification/inference_projects.md)。
