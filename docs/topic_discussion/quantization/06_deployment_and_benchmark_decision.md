# 06. Deployment and Benchmark Decision | 部署与 Benchmark 决策

## 页面目标

这一页负责把前面的量化判断收束到部署和 benchmark：量化是否真的值得切换。

本页的输出是可交付决策：在质量、显存、吞吐、延迟和部署复杂度之间，明确采用、继续调优或回退的理由。

## 问题起点

量化最常见的误判是：显存降了，所以一定值得。工程上真正要问的是：

- 显存和带宽收益是否真实；
- 速度是否真的提升或至少没有明显变差；
- 精度损失是否在目标场景可接受；
- 后端、kernel 和部署复杂度是否也在可控范围。

## 你要先确认什么

- workload 是否固定。
- baseline 和 candidate 是否只改一个关键变量。
- 你的目标更偏精度、显存、吞吐，还是部署成本。

## 为什么这一页必须存在

没有这一页，量化专题就会停在“方法清单”。有了这一页，量化才会回到真正的工程问题：这次压缩有没有带来值得保留的系统收益。

## 判定原则

- `keep`：收益不明显，或者精度 / 部署代价太高。
- `tune`：方向对，但量化粒度、后端、cache policy 或 workload 还要继续调。
- `switch`：收益稳定，并且和目标硬件、服务目标匹配。

## 报告应该怎么写

一个合格的量化报告至少要同时说明：

- 压的是哪一种对象；
- 显存、带宽、TTFT、TPOT、throughput 分别怎么变；
- 精度损失是否可接受；
- backend 和部署复杂度有没有额外代价；
- 最终是继续保留、继续调优，还是换路线。

## 文献与工程入口

- `67` Quantized Inference and Deployment
- `66` Inference Performance Comparison
- Part 03 `10` Triton Quantization

## 典型阅读入口

- [02 PTQ and QAT Timing](./02_ptq_and_qat_timing.md)
- [04 Weight-Only Compression](./04_weight_only_compression.md)
- [05 FP8 and KV Cache Quantization](./05_fp8_and_kv_cache_quantization.md)

## 项目结论

量化路线最终不是靠“名词更先进”成立，而是靠 benchmark 和部署收益成立。

## 回到项目

将结论回填到 `65 QLoRA 选择 -> 66 推理性能比较 -> 67 量化推理与部署`。如果结果只证明模型变小，却没有证明 workload 下的系统收益，应保留为 `tune` 或 `reject`，不要写成已完成优化。
