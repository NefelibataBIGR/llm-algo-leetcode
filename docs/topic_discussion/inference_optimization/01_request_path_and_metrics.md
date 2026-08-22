# 01. Request Path and Metrics | 请求链路与指标口径

## 页面目标

这一页回答两个问题：

- 一个请求从进来到输出 token，链路长什么样？
- TTFT、TPOT、throughput、peak memory 这些指标应该怎么一起看？

## 问题起点

推理优化最常见的误区，不是技巧不够多，而是还没统一“在看什么”。如果 workload 没固定、指标口径没拆开，那么：

- `FlashAttention` 和 `speculative decoding` 的收益没法比较；
- `prefix reuse` 和 `KV cache quantization` 的收益会被混在一起；
- `66` 里的 benchmark report 也会退化成“换了一个配置后看起来更快”。

因此，推理优化的第一步永远不是调 kernel，而是先把请求链路和报告口径定住。

## 你要先确认什么

- workload 是否固定：模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy。
- 是否拆分 prefill 和 decode，而不是只报 total latency。
- 是否同时报告 TTFT、TPOT、throughput 和 peak memory。

## 链路骨架

```text
request
  │
  ▼
tokenize / batch assemble
  │
  ▼
prefill
  │
  ▼
KV cache
  │
  ▼
decode loop
  │
  ▼
detokenize / stream response
  │
  ▼
benchmark report
```

## 为什么这几个指标要一起看

这些指标分别对应推理链路上的不同段落，不应该互相替代：

- `TTFT` 更接近“首 token 要多久出来”，它首先受 prefill、attention kernel 和 prompt 长度影响。
- `TPOT` 更接近“后续每个 token 要多久出来”，它首先受 decode loop、KV cache 和调度影响。
- `throughput` 更接近“系统单位时间能吐多少 token”，它受 batching、调度和策略接受率影响。
- `peak memory` 是预算约束，决定 batch、上下文和 cache policy 能不能继续上去。

如果只看其中一个指标，优化方向很容易走偏。一个典型例子是：throughput 变高了，但 TTFT 也显著变差，这对在线交互往往不是好结果。

## 指标口径

| 指标 | 含义 | 主要关联 |
|:---|:---|:---|
| `TTFT` | Time To First Token，首 token 延迟 | prefill、attention kernel、chunked prefill |
| `TPOT` | Time Per Output Token | decode loop、KV cache 读写、调度 |
| `throughput` | 单位时间生成 token 数 | batching、speculative decoding、多 token 解码 |
| `peak memory` | 推理峰值显存 | 权重、KV cache、batch size、量化 |
| `prefill_share` | prefill 占总耗时比例 | prompt length、attention 访存 |
| `decode_share` | decode 占总耗时比例 | KV cache、sampling、decode scheduling |

## 诊断框架

把一条推理链路压成 4 个问题，会比背优化名词更稳：

1. 这个请求是 `prefill-bound` 还是 `decode-bound`？
2. 如果显存接近预算，它是不是已经变成 `memory-bound`？
3. 当前收益应该优先来自 kernel、策略、缓存管理还是量化？
4. 报告里是否能用同一 workload 证明这次变化值得保留？

可以用一个简化判断表快速分流：

| 信号 | 更可能的问题 | 下一页先看什么 |
|:---|:---|:---|
| `TTFT` 高、长 prompt 一拉长就慢 | `prefill-bound` | `02` |
| `TPOT` 高、并发时 token 吐得慢 | `decode-bound` | `03` |
| peak memory 顶到预算、batch 上不去 | `memory-bound` | `04` + `05` |
| 没有明显单点瓶颈 | 需要回到端到端判断 | `06` |

![Inference request lifecycle](/topic_discussion/inference_optimization/request_lifecycle.svg)

## 与 Part 02 Task1-6 的关系

这页不是简单复述 `Task1-6`。它承担的是“知识组织层”的入口作用：

- `Task1-6` 负责学习顺序，告诉读者该先读哪些 notebook；
- `01` 负责把这些 notebook 放回同一条请求链路里，告诉读者“为什么要分 prefill、decode、KV cache、量化这几条线”；
- 因此，这一页更像诊断起点，而不是文件索引。

## 文献锚点

- Vaswani et al., *Attention Is All You Need*：给出 decoder 自回归推理的基础形态。
- Dao et al., *FlashAttention*：帮助理解为什么 attention 不是纯 FLOPs 问题，而是访存问题。
- Kwon et al., *vLLM / PagedAttention*：帮助理解服务系统为什么必须重新设计 cache 管理方式。

## 常见误区

- 只看 throughput，不看 TTFT。
- 不拆 prefill / decode，只报 total latency。
- workload 没固定，就比较优化结果。
- 只看单条请求，不看请求分布。

## 对应 Part 02

- `20` FlashAttention Sim
- `21` Decoding Strategies
- `22` vLLM PagedAttention
- `23` Speculative Decoding
- `24` SGLang RadixAttention
- `25 / 40 / 41 / 67` 量化推理与部署
- `66` Inference Performance Comparison

## 典型阅读入口

- [06 Benchmark and Decision](./06_benchmark_and_decision.md)
- [02 Prefill and Attention Kernel](./02_prefill_and_attention_kernel.md)
- [03 Decoding Strategies](./03_decoding_strategies.md)
- [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.md)

## 本节要点

没有统一的 workload 和指标口径，后面的推理优化都没有可比性。
