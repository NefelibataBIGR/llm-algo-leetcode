# 04. KV Cache and Scheduling | KV Cache 与调度

## 页面目标

这一页回答的是：KV cache 怎么增长、复用、分页、驱逐，decode 请求怎么排。

## 问题起点

只要系统开始做长上下文、多轮对话或高并发服务，KV cache 就很快从“实现细节”变成“系统边界”：

- cache 会随着层数、上下文长度和 batch 线性增长；
- 一旦 cache 顶到预算，batch、上下文和并发都上不去；
- 就算还没 OOM，碎片、分页和调度也会直接拖慢 TPOT。

这就是为什么 `KV cache` 会同时出现在推理优化和显存优化专题里，但两边看的目标不同。

## 你要先确认什么

- peak memory 是否接近预算。
- 长上下文和并发请求是否把 cache 撑爆。
- prefix reuse 是否有明显收益。

## 核心矛盾

KV cache 的核心矛盾是：它既是 decode 提速所需的缓存，又是推理侧最稳定增长的显存对象。系统既希望尽量保留更多上下文，又希望不要因为 cache 组织方式把吞吐和预算一起拖垮。

## 演化路径

KV cache 是推理链路里最容易成为硬约束的部分。

1. cache 会随层数、head 数、长度和 batch 持续增长。
2. prefix caching 让重复前缀尽量复用。
3. PagedAttention 把连续缓存变成块管理。
4. RadixAttention 让前缀树式复用更高效。
5. decode scheduling 决定请求怎样错峰和排序。

## 关键取舍

- `prefix caching` 更适合重复前缀明显的 workload，不是所有请求都会收益。
- `PagedAttention` 改的是 cache 管理粒度，收益常体现在碎片和并发上。
- `RadixAttention` 更强调前缀共享和树式组织，但也要求请求模式与系统实现匹配。
- `KV cache quantization` 能继续压预算，但不应替代复用、分页和调度本身。

因此，这一页的读法应该是：先看 cache 是否成为硬约束，再决定先做复用、分页、调度还是压缩。

![KV cache lifecycle and scheduling](/topic_discussion/inference_optimization/kv_cache_scheduling.svg)

## 文献锚点

- Kwon et al., *vLLM / PagedAttention*：理解为什么服务系统必须把 cache 当块管理。
- SGLang / RadixAttention 相关资料：理解前缀树式复用的系统收益。
- prefix caching / chunked prefill 工程资料：理解复用与 prefill 分块的协同关系。

## 常见误区

- 把 KV cache 当成纯实现细节，不看它的增长曲线。
- 只看单请求，不看并发。
- 看到 cache 占用高就直接量化，不先看复用和调度。

## 对应 Part02

- `22` vLLM PagedAttention
- `24` SGLang RadixAttention
- `34` Prefix Caching and Chunked Prefill
- `37` KV Cache Scheduling
- `41` FP8 and KV Cache Quantization

## 经典阅读入口

- [11 KV Cache and Memory Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb)
- [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
- [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)
- [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)
- [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)

## 相关跳转

- 看 `01`，确认指标口径。
- 看 `03`，确认 decode 循环怎么耗时。
- 看 `05`，确认 cache 不够时怎么压缩。

## 小结

KV cache 不是附属缓存，而是推理吞吐和上下文长度的核心边界。
