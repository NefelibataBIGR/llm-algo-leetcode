# 量化与压缩专题

## 专题定位

本专题用于串起量化主线：先看低比特表示到底在压什么，再看 PTQ、QAT、GPTQ、AWQ、FP8、KV cache quant 分别在什么时机介入，最后把收益收回推理、显存和部署约束里的最终选型。这里聚焦量化方法轴；如果问题已经明确是推理速度或显存预算，应转到对应专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part01 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 量化基础与硬件直觉 | `P1:01 -> 12 -> 21` | [01 Quantization Object and Error](./01_quantization_object_and_error.md) |
| Task2 | PTQ / QAT 的介入时机 | `21 -> 25 -> 26` | [02 PTQ and QAT Timing](./02_ptq_and_qat_timing.md) |
| Task3 | GPTQ / AWQ 的后训练压缩 | `25 -> 40` | [04 Weight-Only Compression](./04_weight_only_compression.md) |
| Task4 | FP8 与 KV cache quant | `P1:03 -> 12 -> 41` | [05 FP8 and KV Cache Quantization](./05_fp8_and_kv_cache_quantization.md) |
| Task5 | 量化部署与服务选型 | `67` | [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md) |
| Task6 | benchmark 与项目收口 | `66` | [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“到底该压权重、激活还是 KV cache”“PTQ 和 QAT 哪条路更适合当前约束”时，再回来看对应的专题正文。想看汇总版就进 [量化与压缩正文](./casebook.md)，想按连续故事线走一遍就进 [量化与压缩深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[推理优化专题](../inference_optimization/intro.md) 负责服务速度与部署链路，[显存优化专题](../memory_performance_tuning/intro.md) 负责 VRAM 压缩 trade-off，[监督微调专题](../fine_tuning_training/intro.md) 负责 QLoRA 等训练时机问题。
