# 05. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化

## 页面目标

这一页回答的是：FP8 和 KV cache quantization 为什么总被放在一起讨论，以及它们到底改的是哪一类成本。

## 问题起点

这两条路线容易被误解成“又一种更低比特”。但它们的意义并不在于位宽本身，而在于：

- FP8 更像硬件与执行栈驱动的低精度路径；
- KV cache quantization 更像推理预算驱动的缓存压缩路径。

它们都和传统 weight-only 路线不同。

## 你要先确认什么

- 当前瓶颈来自执行栈，还是来自 cache 预算。
- 硬件是否已经原生支持 FP8。
- 长上下文和并发是否已经把 KV cache 顶成第一约束。

## 核心矛盾

FP8 的核心矛盾是：更低精度的执行路径能带来更好的吞吐和存储效果，但要求硬件和 kernel 栈配合；KV cache quantization 的核心矛盾是：缓存压缩能扩大上下文和并发预算，但会影响表示精度和服务稳定性。

## 演化路径

1. 先判断问题是在执行路径还是缓存预算。
2. 如果是执行栈和硬件支持，优先看 FP8。
3. 如果是长上下文和高并发预算，优先看 KV cache quantization。
4. 最后再回到推理和显存专题，看它们在请求链路和预算中的综合效果。

## 关键取舍

- FP8 更依赖硬件和 backend 的成熟度。
- KV cache quantization 更依赖 workload 的上下文长度和并发特征。
- 二者都不能只看理论压缩率，必须回到服务目标和 benchmark。

![FP8 and KV cache quantization](/topic_discussion/quantization/fp8_kv_cache.svg)

## 文献锚点

- FP8 相关资料：理解低精度执行路径如何和硬件协同。
- KV cache quantization 资料：理解缓存压缩为什么首先是推理预算问题。

## 对应 Part02

- `41` FP8 and KV Cache Quantization
- `67` Quantized Inference and Deployment

## 典型阅读入口

- [04 Weight-Only Compression](./04_weight_only_compression.md)
- [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md)

## 小结

FP8 和 KV cache quantization 都是低精度路线，但一个更偏执行栈，一个更偏缓存预算。
