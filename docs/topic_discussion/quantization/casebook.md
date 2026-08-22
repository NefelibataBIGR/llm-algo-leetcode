# 量化与压缩正文

这页只做量化问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先判断压缩对象，再判断介入时机；随后区分训练适配、权重量化和推理侧量化，最后用部署 benchmark 验证。不要从算法名倒推问题，也不要把模型文件变小直接当成上线理由。

## 判断表

先分清问题出在权重、激活还是 KV cache，再判断量化应该发生在训练后还是训练中，最后再看部署与 benchmark 是否支持这条路线。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 模型太大，先想快速压缩 | `PTQ entry` | [02](./02_ptq_and_qat_timing.md), [04](./04_weight_only_compression.md) | 先做 PTQ / W8A16 验证收益 |
| PTQ 之后误差太大 | `QAT / adaptation` | [02](./02_ptq_and_qat_timing.md), [03](./03_low_bit_training_adaptation.md) | 看 QAT、QLoRA 或训练适配路线 |
| 想保住低比特精度 | `post-training compensation` | [04](./04_weight_only_compression.md) | 看 GPTQ / AWQ 的误差补偿 |
| 长上下文服务被 cache 顶住 | `cache quant` | [05](./05_fp8_and_kv_cache_quantization.md) | 看 FP8 / KV cache quant |
| 量化结果能跑，但值不值得上线 | `deployment validation` | [06](./06_deployment_and_benchmark_decision.md) | 回到 workload、精度、吞吐和部署成本 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 量化对象 | 压的是权重、激活还是 KV cache | 量化就是统一改 dtype |
| 介入时机 | 是 PTQ、QAT 还是低比特训练适配 | 先看流行度，不看约束 |
| 后训练补偿 | GPTQ / AWQ 是否真在保精度 | 只看 bit 数，不看误差路径 |
| 部署验证 | 省出来的显存或带宽值不值 | 显存降了就默认 adopt |

## 本节要点

这页的职责不是再讲一遍量化术语，而是把量化选型里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小决策模板

记录 `对象 -> 误差边界 -> 介入时机 -> 后端约束 -> workload 指标 -> 决策`。至少同时保留质量、显存、吞吐/延迟和部署复杂度四类证据。
