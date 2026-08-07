# 推理优化专题

## 专题概览

本专题用于沉淀大模型推理优化路线，回答三个问题：

- 一个请求从进来到输出 token，主要慢在哪里？
- FlashAttention、解码策略、KV cache、调度和量化分别改的是哪一段？
- 如何把优化结果落到 `66_Inference_Performance_Comparison.ipynb` 的 benchmark report 里？

当前策略是：**横向专题负责完整链路，非项目 notebook 负责单点机制，`66` 负责项目收口**。这样可以减少对已经成形的非项目节的重复修改，也方便在专题页里加入更多路线链接。

## 职责边界

这个专题只负责推理链路里的性能优化、缓存管理、调度和部署侧选型，不负责训练流程本身，也不替代 profiling 专题。

- `FlashAttention` 关注 attention 计算里的 HBM/SRAM 访存瓶颈。
- `Decoding` 关注采样、搜索、推测解码和多 token 生成。
- `KV Cache` 关注缓存增长、分页、复用、驱逐和调度。
- `Serving Scheduling` 关注 prefill / decode / batch / priority 如何排布。
- `Quantized Deployment` 关注权重、FP8 和 KV cache 量化在部署侧的收益与代价。
- `66` 负责把这些策略放进同一 workload，输出 `keep / tune / switch` 选型结论。

和 `显存优化与性能调优专题` 的区别是：

- 本专题先看请求链路是否更快、更顺，重点是 `TTFT / TPOT / throughput`。
- 显存专题先看资源是否被压进预算，重点是 `peak memory / VRAM ledger / trade-off`。
- 两边都会碰到 `KV cache` 和量化，但本专题把它们当成推理链路上的调度与吞吐问题来看。

## 推理链路总图

```text
request
  │
  ▼
tokenize / batch assemble
  │
  ▼
prefill ── attention kernel / FlashAttention / chunked prefill
  │
  ▼
KV cache ── allocation / paging / prefix reuse / eviction
  │
  ▼
decode loop ── sampling / speculative decoding / multi-token decoding / scheduling
  │
  ▼
detokenize / stream response
  │
  ▼
benchmark report ── TTFT / TPOT / throughput / peak memory / decision
```

## Task1-6 主线

| Task | 主题 | 推荐小节 | 学完应能回答 |
|:---|:---|:---|:---|
| Task1 | 推理结构地基 | [04 Attention MHA/GQA](../../02_PyTorch_Algorithms/04_Attention_MHA_GQA.ipynb)、[05 LLaMA3 Block](../../02_PyTorch_Algorithms/05_LLaMA3_Block_Tutorial.ipynb) | 推理时 attention、GQA 和 block 数据流主要消耗在哪里？ |
| Task2 | Attention 访存瓶颈与 FlashAttention | Part01 [03 GPU Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)、[14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)、[24 SRAM Optimization](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb) + Part02 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) | 为什么 attention 优化不是只看 FLOPs，而是看 HBM/SRAM、tiling 和中间矩阵读写？ |
| Task3 | 解码算法与生成策略 | [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.ipynb)、[23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.ipynb)、[35 Multi-Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.ipynb) | token 怎么生成，如何减少 decode 循环成本？ |
| Task4 | KV Cache、Prefix 复用与 Decode 调度 | Part01 [11 KV Cache Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) + Part02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)、[24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)、[34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)、[36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) | KV cache 怎么增长、复用、分页、驱逐，decode 请求怎么排？ |
| Task5 | 量化推理与部署 | Part01 [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) + Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[40 GPTQ and AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | 权重、激活和 KV cache 量化分别改变什么成本？ |
| Task6 | 推理性能对比项目 | [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) | 如何用同一 workload 比较 TTFT、TPOT、吞吐、显存并输出选型？ |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `2.6` | FlashAttention、Decoding、PagedAttention 的基础入口 | [2.6 核心推理优化](../../02_PyTorch_Algorithms/2_6.md) |
| `2.7A` | Prefix caching、multi-token decoding、decode scheduling 的高级入口 | [2.7A 高级推理策略](../../02_PyTorch_Algorithms/2_7A.md) |
| `2.9` | 项目收口组入口 | [2.9 综合项目与性能对比](../../02_PyTorch_Algorithms/2_9.md) |
| `66` | 推理性能对比项目，输出 `keep / tune / switch` | [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) |

## 推荐入口

- 如果你从零学推理优化，按 Task1-6 顺序读。
- 如果你只关心 attention kernel，从 Task2 开始，再回到 `66` 看收益如何验证。
- 如果你只关心服务吞吐，从 Task4 开始，再接 Task6。
- 如果你只关心部署压缩，从 Task5 开始，再接 Task6。

## 正文页

- [推理优化正文](./casebook.md)：指标口径、瓶颈诊断、路线对照和常见误区。
- [推理优化深入阅读](./walkthrough.md)：从一个请求进入系统开始，连续走到 `66` 的 benchmark report。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先证明慢在哪里、用什么指标支撑结论时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当问题集中在 KV cache、显存预算和吞吐取舍时看这里。
- [量化与压缩专题](../quantization/intro.md)：当问题集中在 W8A16、GPTQ/AWQ、FP8 或 KV cache quant 时看这里。
- [编译与图优化专题](../compiler_graph_optimization/intro.md)：当优化更像 backend、fusion 或 kernel schedule 差异时看这里。

## 专题状态

本专题已更新为新版推理优化主线入口。后续优先维护 `casebook.md / walkthrough.md`，尽量减少对非项目 notebook 的重复扩写。
