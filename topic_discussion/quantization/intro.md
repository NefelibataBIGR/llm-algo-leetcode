# 量化与压缩专题

## 专题定位

本专题用于串起量化主线：先看低比特表示到底在压什么，再看 PTQ、QAT、GPTQ、AWQ、FP8、KV cache quant 分别在什么时机介入，最后把收益收回推理、显存和部署约束里的最终选型。这里聚焦量化方法轴；如果问题已经明确是推理速度或显存预算，应转到对应专题。

## Infra 层定位

量化专题主要落在 L2-L4：L2 关心低比特算子、kernel 和硬件支持，L3 关心训练/校准/量化配置，L4 关心量化模型在推理引擎中的加载、KV cache 与服务吞吐。模型权重、激活和 KV cache 是被压缩的负载，不单独构成一层；部署成本还要回到 L1 的显存容量与带宽、以及 L5 的评测和资源约束。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的量化与部署路线进入，再用 [Part 02 资产表](../../02_PyTorch_Algorithms/2_10.md) 定位 65、66、67 等项目节。量化专题是方法选择支撑线，可以按当前约束切入，不必从 PTQ 到部署完整顺读。

## 前置阅读

建议先掌握 `Part 01: 21` 的量化理论与 INT4/INT8 基础，再根据目标补读 `Part 01: 25`、`26`、`40`、`41`。如果重点是训练显存，应同时回看监督微调与显存专题；如果重点是服务吞吐，应先明确推理后端和 workload。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 量化基础与硬件直觉 | `Part 01:01 -> 12 -> 21` | [01 Quantization Object and Error](./01_quantization_object_and_error.md) |
| Task2 | PTQ / QAT 的介入时机 | `21 -> 25 -> 26` | [02 PTQ and QAT Timing](./02_ptq_and_qat_timing.md) |
| Task3 | 低比特训练与适配 | `26` | [03 Low-Bit Training Adaptation](./03_low_bit_training_adaptation.md) |
| Task4 | GPTQ / AWQ 的后训练压缩 | `25 -> 40` | [04 Weight-Only Compression](./04_weight_only_compression.md) |
| Task5 | FP8 与 KV cache quant | `Part 01:03 -> 12 -> 41` | [05 FP8 and KV Cache Quantization](./05_fp8_and_kv_cache_quantization.md) |
| Task6 | 量化部署、benchmark 与项目收口 | `65 -> 66 -> 67` | [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走来源主线；遇到“到底该压权重、激活还是 KV cache”“PTQ 和 QAT 哪条路更适合当前约束”时，再回来看对应的专题正文。想看汇总版就进 [量化与压缩正文](./casebook.md)，想按连续故事线走一遍就进 [量化与压缩深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[推理优化专题](../inference_optimization/intro.md) 负责服务速度与部署链路，[显存优化专题](../memory_performance_tuning/intro.md) 负责 VRAM 压缩 trade-off，[监督微调专题](../fine_tuning_training/intro.md) 负责 QLoRA 等训练时机问题。

## 项目结论

推荐的实践闭环是 `65 QLoRA 选择 -> 66 推理性能比较 -> 67 量化推理与部署`；若要比较具体权重量化方法，可继续接入 `40 GPTQ and AWQ Weight Quantization`。最终结论应同时报告精度或任务质量、显存占用、吞吐或延迟，以及模型格式和后端约束。

## 环境与验证

量化理论、误差计算和部分 W8A16 模拟可先用 CPU；真实权重量化、GPU 推理和后端部署通常需要 GPU。不同显卡、驱动、PyTorch、量化库和 serving backend 可能改变结果，必须记录环境与校准数据，并保存可复现的配置和结果文件。
