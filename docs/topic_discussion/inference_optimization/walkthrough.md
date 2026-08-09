# 推理优化深入阅读

## 主故事线

把推理优化看成一条请求链路，会比按名词背技巧更稳：

`request -> tokenize / batch assemble -> prefill -> KV cache -> decode loop -> detokenize -> benchmark report`

每个优化方法都应该落在这条链路里的某一段。FlashAttention 主要处理 prefill 和 attention 的访存问题；PagedAttention、RadixAttention、prefix caching 和 KV cache scheduling 主要处理缓存管理；speculative decoding、multi-token decoding 和 decode scheduling 主要处理生成阶段吞吐；量化推理主要处理权重、带宽和 KV cache 存储成本；`66` 则负责把所有候选方案放回同一个 workload 做对比。

如果你已经知道自己的问题落在哪一段，可以直接跳到对应编号页：

- [01 Request Path and Metrics](./01_request_path_and_metrics.md)
- [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md)
- [03 Decoding Strategies](./03_decoding_strategies.md)
- [04 KV Cache and Scheduling](./04_kv_cache_and_scheduling.md)
- [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md)
- [06 Benchmark and Decision](./06_benchmark_and_decision.md)

## 01 请求进入系统

一个真实请求进来时，第一步不是优化，而是固定 workload：

- 模型和 backend 是什么
- batch size 是多少
- prompt tokens 和 generated tokens 多长
- dtype 是 FP16、INT8、FP8 还是其他
- KV cache 是静态、分页、复用还是量化

这对应 `66` 里的 `build_inference_config`。如果 workload 没有固定，后面的 TTFT、TPOT、throughput 和 peak memory 都不能比较。

## 02 Prefill 先把 prompt 写进上下文

prefill 阶段会处理已有 prompt。长 prompt 让 attention 计算和中间 score 矩阵压力上升，这时先看：

- Part01 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- Part01 [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.md)
- Part01 [24 SRAM Optimization Techniques](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.md)
- Part02 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.md)

如果 `66` 里 `TTFT` 高、`prefill_share` 高，优先从这里找原因。

## 03 Decode Loop 决定持续生成速度

decode 阶段是一轮一轮生成 token。这里的关键不是只看“怎么采样”，还要看每轮是否有足够高的利用率，以及请求是否被排顺。

这一段建议按这个顺序看：

- Part02 [21 Decoding Strategies](../../02_PyTorch_Algorithms/21_Decoding_Strategies.md)
- Part02 [23 Speculative Decoding](../../02_PyTorch_Algorithms/23_Speculative_Decoding.md)
- Part02 [35 Multi-Token Decoding](../../02_PyTorch_Algorithms/35_Multi_Token_Decoding.md)
- Part02 [36 Decode Scheduling](../../02_PyTorch_Algorithms/36_Decode_Scheduling.md)

如果 `66` 里 `TPOT` 高、`decode_share` 高，就说明问题主要在 decode 阶段。

## 04 KV Cache 决定长上下文和并发边界

prefill 结束后，模型会留下 KV cache。后续 decode 每生成一个 token，都会继续读写这些缓存。cache 的成本和层数、batch、序列长度、KV head 数、dtype 都有关。

这一段建议按这个顺序看：

- Part01 [11 KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- Part02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.md)
- Part02 [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.md)
- Part02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.md)
- Part02 [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.md)

如果 peak memory 接近预算，`66` 应优先把瓶颈判成 `memory-bound`。

## 05 量化推理处理部署成本

当显存、带宽或部署成本成为主要约束时，再进入量化推理线：

- Part01 [21 Quantization Theory and INT4 INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- Part02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)
- Part02 [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)
- Part02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)

量化的判断要回到服务目标：在线交互更敏感 TTFT / TPOT，离线批处理更敏感 throughput / cost。

## 06 回到 66 做项目收口

最后把候选方案放进 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)。

`66` 当前的项目闭环是：

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

一个合格的推理优化结论至少应该写清楚：

- baseline 和 candidate 的 workload 是否一致
- TTFT、TPOT、throughput 和 peak memory 分别怎么变
- 主要瓶颈是 prefill、decode、memory 还是 balanced
- candidate 的收益是否匹配目标场景
- 最终是 `keep`、`tune` 还是 `switch`

## 典型路径

### 长 prompt 首 token 慢

`02 -> 04 -> 06`

先看 FlashAttention 和 SRAM/HBM 访存，再看 prefix caching 和 chunked prefill，最后用 `66` 的 TTFT 和 prefill_share 验证。

### 并发高时吞吐低

`03 -> 04 -> 06`

先看 decoding 和 multi-token 生成，再看 decode scheduling 和 KV cache scheduling，最后用 throughput_gain、TPOT delta 和 TTFT delta 判断是否切换。

### 显存卡住 batch 和上下文

`04 -> 05 -> 06`

先看 KV cache 增长、分页、复用和驱逐，再看 KV cache quantization，最后用 peak_mem_delta 和 TPOT/TTFT 退化判断取舍。

## 阅读建议

- 想按完整路线学习，先回到 [推理优化专题入口](./intro.md)。
- 想查指标和误区，回到 [推理优化正文](./casebook.md)。
- 想做项目收口，直接进入 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)。
