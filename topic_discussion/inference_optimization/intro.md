# 推理优化专题

## 专题概览

本专题用于沉淀大模型推理优化路线，回答三个问题：

- 一个请求从进来到输出 token，主要慢在哪里？
- FlashAttention、解码策略、KV cache、调度和量化分别改的是哪一段？
- 如何把优化结果落到 `66_Inference_Performance_Comparison.ipynb` 的 benchmark report 里？

当前策略是：**Part02 的 Task1-6 负责主学习线，横向专题负责给这条主线补解释层、判断框架和图册，`66` 负责项目收口**。这样可以减少对已经成形的非项目节的重复修改，也方便在专题页里加入更多路线链接。

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

## 01-06 骨架

这 6 个小节是知识组织层，不要求和 Part02 的 Task1-6 一一对应。它们围绕“一个请求怎么变快”来组织，读者可以按问题入口自由跳读。

| 章节 | 你会得到什么 | 适合先从哪里进入 |
|:---|:---|:---|
| `01` | 请求路径、指标口径和诊断框架 | 先想弄清楚该看 TTFT、TPOT、throughput 还是 peak memory |
| `02` | prefill、attention kernel、FlashAttention | 长 prompt 变慢，先看这一页 |
| `03` | decoding、speculative、multi-token | decode 循环太多，先看这一页 |
| `04` | KV cache、prefix reuse、paging、scheduling | cache 涨、并发高、请求排不顺，先看这一页 |
| `05` | W8A16、GPTQ/AWQ、FP8、KV cache quant | 显存或带宽受限，先看这一页 |
| `06` | 端到端 benchmark、选型结论 | 已经有候选方案，直接看这一页 |

`66` 继续作为项目收口页，用来把这些判断放回同一个 workload。

## Task1-6 与 01-06 的关系

这里保留两套结构，它们不冲突，职责也不同：

- `Task1-6` 是学习内容的组织方式，回答“先学什么、后学什么、先补哪段 notebook”。
- `01-06` 是知识组织层，回答“这个问题的全貌是什么、几类方案分别解决哪一段瓶颈”。
- 同一个 `Task` 可以引用多个 `01-06` 页面；同一个 `01-06` 页面也可以服务多个 `Task`。
- 因此，`01-06` 不能退化成文件索引，它必须承担叙事骨架、文献锚点和可视化入口。

可以用下面这张关系表来理解：

| 维度 | 作用 | 典型问题 |
|:---|:---|:---|
| `Task1-6` | 学习路径 | 这一轮应该先补哪组 notebook？ |
| `01-06` | 知识组织 | 这个瓶颈为什么存在，常见解法如何分层？ |
| `66` | 项目收口 | 哪套策略值得继续保留、调优或切换？ |

## 与显存优化专题的重合点

`推理优化` 和 `显存优化与性能调优` 会共享一批机制，但两边的问题视角不同。重合不代表重复，关键在于先看哪个目标。

| 重合内容 | 本专题先回答什么 | 显存专题先回答什么 |
|:---|:---|:---|
| `KV cache` | 为什么影响 `TTFT / TPOT / throughput`，如何通过 paging、reuse、scheduling 让请求更快 | 为什么把 `peak memory` 顶高，如何通过预算、分页、压缩把 cache 压进显存 |
| `量化` | 是否让部署链路更快、更顺，是否值得切换后端或精度 | 是否显著降低 VRAM / 带宽压力，代价是否可接受 |
| `benchmark / profiling` | 哪段请求链路最慢，优化后延迟和吞吐提升多少 | 哪个资源对象最占显存，时间换空间是否划算 |
| `调度` | batch、prefill、decode 如何排队才能提高服务吞吐 | 资源排布是否把 cache、activation 或 buffer 顶到预算外 |

一个简单判断原则是：

- 如果你在问“为什么慢”，先看本专题。
- 如果你在问“为什么装不下”，先看显存专题。
- 如果你在问“既慢又占显存”，通常先从 `KV cache`、量化和调度这三条交叉线切入。

## 文献锚点

每一页都应保留 3-5 个可继续深挖的起点，避免横向专题退化成纯索引。

- `02 Prefill and Attention Kernel`：FlashAttention、chunked prefill、HBM/SRAM 访存模型。
- `03 Decoding Strategies`：speculative decoding、multi-token decoding、decode loop 优化。
- `04 KV Cache and Scheduling`：vLLM PagedAttention、SGLang RadixAttention、prefix caching。
- `05 Quantized Inference and Deployment`：W8A16、GPTQ、AWQ、FP8、KV cache quantization。
- `06 Benchmark and Decision`：同一 workload 下的 TTFT、TPOT、throughput、peak memory 选型。

## 推荐入口

- 如果你还没建立指标口径，先看 `01`。
- 如果你先遇到长 prompt / 首 token 慢，先看 `02`。
- 如果你先遇到生成慢，先看 `03`。
- 如果你先遇到 cache 和并发问题，先看 `04`。
- 如果你先遇到显存或部署成本，先看 `05`。
- 如果你已经有候选方案，直接看 `06` 和 `66`。

## 正文页

- [01 Request Path and Metrics](./01_request_path_and_metrics.md)
- [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md)
- [03 Decoding Strategies](./03_decoding_strategies.md)
- [04 KV Cache and Scheduling](./04_kv_cache_and_scheduling.md)
- [05 Quantized Inference and Deployment](./05_quantized_inference_and_deployment.md)
- [06 Benchmark and Decision](./06_benchmark_and_decision.md)
- [07 Visual Assets](./07_visual_assets.md)
- [推理优化正文](./casebook.md)：指标口径、瓶颈诊断、路线对照和常见误区。
- [推理优化深入阅读](./walkthrough.md)：从一个请求进入系统开始，连续走到 `66` 的 benchmark report。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先证明慢在哪里、用什么指标支撑结论时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当问题集中在 KV cache、显存预算和吞吐取舍时看这里。
- [量化与压缩专题](../quantization/intro.md)：当问题集中在 W8A16、GPTQ/AWQ、FP8 或 KV cache quant 时看这里。
- [编译与图优化专题](../compiler_graph_optimization/intro.md)：当优化更像 backend、fusion 或 kernel schedule 差异时看这里。

## 专题状态

本专题已更新为 `01-06 + 07_visual_assets` 的解释层结构。它的作用是把推理优化这件事组织成问题驱动的知识切面，而不是把 Part02 目录重排一遍。`66` 继续作为项目收口页。
