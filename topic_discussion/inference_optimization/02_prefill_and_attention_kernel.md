# 02. Prefill and Attention Kernel | Prefill 与 Attention Kernel

## 页面目标

这一页回答的是：长 prompt 为什么会慢，FlashAttention 和 chunked prefill 具体改的是哪一段。

## 问题起点

推理链路里，首 token 延迟往往最先暴露出 prefill 的代价。用户感受到的是“输入一大段上下文后，模型迟迟不出第一个 token”，但真正的问题常常不是模型参数量本身，而是：

- prompt 太长导致 attention 访存和中间写回膨胀；
- prefill 把大量已有 token 一次性送进模型，导致 `TTFT` 被这一段主导；
- backend 还在用对短 prompt 友好的实现，遇到长 prompt 就开始掉速。

## 你要先确认什么

- TTFT 是否在长 prompt 下明显升高。
- `prefill_share` 是否高于 decode。
- attention 是否被中间 score 矩阵和 HBM 读写拖慢。

## 核心矛盾

prefill 的核心矛盾不是“算力够不够”，而是“访存和中间结果要不要反复写回 HBM”。长上下文下，attention 的理论复杂度大家都知道，但真正把 TTFT 顶高的，往往是中间 score、softmax 和 value 聚合带来的内存路径。

## 演化路径

prefill 不是“先算一遍前向”这么简单。它要把已有 prompt 组织成上下文，同时完成 attention 计算。

1. prompt 变长后，中间矩阵和带宽压力上升。
2. naive attention 往往被 HBM 读写拖慢。
3. FlashAttention 通过 tiling 和 online softmax 减少中间写回。
4. chunked prefill 进一步把长 prompt 分块处理。
5. 最终目标是把 TTFT 压下来，而不是只看 FLOPs。

## 关键取舍

这条线的 trade-off 很明确：

- `FlashAttention` 主要换来更好的访存路径，但要求 kernel 和 backend 更匹配；
- `chunked prefill` 主要解决超长 prompt 的工程落地问题，但会改变调度和 cache 的接入方式；
- `prefix caching` 可以减少重复 prefill，但它解决的是“重复前缀”而不是“所有长 prompt 都慢”。

因此，看到 TTFT 高时，不能把这三者混成一个动作，它们处理的是不同层面的瓶颈。

![Prefill and attention kernel](/topic_discussion/inference_optimization/prefill_attention.svg)

## 文献锚点

- Dao et al., *FlashAttention*：理解 online softmax 和 tiling 为何能显著减少 HBM 写回。
- Dao, *FlashAttention-2*：关注并行分工和 kernel 落地如何继续改进吞吐。
- chunked prefill 相关工程资料：帮助理解长 prompt 在服务系统里的分块处理方式。

## 常见误区

- 只看 FLOPs，忽略 HBM/SRAM 访存。
- 把 prefill 慢简单等同于模型本身慢。
- chunked prefill 和 prefix caching 混为一谈。

## 对应 Part 02

- `20` FlashAttention Sim
- `34` Prefix Caching and Chunked Prefill
- `66` Inference Performance Comparison

## 经典阅读入口

- [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)
- [14 FlashAttention Memory Model](../../01_Hardware_Math_and_Systems/14_FlashAttention_Memory_Model.ipynb)
- [24 SRAM Optimization Techniques](../../01_Hardware_Math_and_Systems/24_SRAM_Optimization_Techniques.ipynb)
- [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb)

## 相关跳转

- 看 `01`，确认指标口径。
- 看 `04`，确认 prefill 结束后 cache 怎么接。

## 本节要点

prefill 优化的重点是减少访存和中间写回，把首 token 延迟压下来。
