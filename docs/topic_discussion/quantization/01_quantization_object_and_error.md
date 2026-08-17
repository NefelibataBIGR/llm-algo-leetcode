# 01. Quantization Object and Error | 量化对象与误差直觉

## 页面目标

这一页回答两个问题：

- 量化到底在压什么，压的是权重、激活，还是 KV cache？
- 为什么量化的核心不是“位宽更小”，而是“误差能不能被系统接受”？

## 问题起点

量化最容易被误解成“把 `fp16` 改成 `int8`”。但真正决定量化效果的，不只是位宽，而是三件事：

- 你压缩的是哪一类对象；
- 你用什么 scale / zero-point / 粒度去表示它；
- 量化误差会不会落在系统最敏感的位置。

如果这三件事没分清，后面的 PTQ、QAT、GPTQ、AWQ、FP8 都会变成名词堆叠。

## 你要先确认什么

- 当前要压的是权重、激活，还是 KV cache。
- 你的目标是省显存、降带宽，还是配合硬件执行栈。
- 误差更敏感的是精度，还是服务侧吞吐和延迟。

## 核心矛盾

量化的核心矛盾是：低比特表示能显著降低存储和带宽成本，但也会引入表示误差。量化专题的主线，就是在“压缩率”和“误差可接受性”之间找平衡。

## 演化路径

1. 先识别压缩对象：权重、激活、KV cache。
2. 再识别量化粒度：per-tensor、per-channel、分组量化。
3. 再判断误差是否会放大到输出质量、训练稳定性或服务指标上。
4. 最后才决定走 PTQ、QAT、GPTQ / AWQ、FP8 或 cache quant。

## 关键取舍

- 权重量化首先影响驻留大小和带宽。
- 激活量化更容易碰到执行路径和精度稳定性问题。
- KV cache 量化更偏推理预算，不应与权重量化混为一谈。

![Quantization objects and error routes](/topic_discussion/quantization/quantization_objects.svg)

## 文献锚点

- 量化基础资料：对称 / 非对称量化、scale、zero-point、误差模型。
- 量化硬件资料：低比特表示为什么会同时影响访存、吞吐和部署选型。

## 对应 Part02

- `25` Quantization W8A16
- `40` GPTQ and AWQ Weight Quantization
- `41` FP8 and KV Cache Quantization
- `67` Quantized Inference and Deployment

## 典型阅读入口

- [02 PTQ and QAT Timing](./02_ptq_and_qat_timing.md)
- [04 Weight-Only Compression](./04_weight_only_compression.md)
- [05 FP8 and KV Cache Quantization](./05_fp8_and_kv_cache_quantization.md)

## 小结

量化的第一步不是选算法，而是先分清压缩对象、误差来源和目标约束。
