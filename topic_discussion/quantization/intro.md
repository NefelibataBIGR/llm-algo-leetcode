# 量化与压缩专题

## 专题概览

本专题用于把模型量化、低比特压缩和量化部署串成一条独立学习路线，回答三个问题：

- 量化到底在压什么，压的是权重、激活，还是 KV cache？
- PTQ、QAT、GPTQ、AWQ、FP8 分别在什么环节起作用？
- 怎么把量化结果放回推理、显存和部署约束里做选型？

这个专题不是推理优化的附属页，而是一个独立方法轴。它和推理优化、显存优化、算子优化都交叉，但关注点更集中在“表示精度、压缩率和部署选型”。

## 职责边界

这个专题负责量化理论、低比特表示、量化训练和量化部署选型，不负责 FlashAttention、decode scheduling 或一般性的 serving 调度。

- `Quantization Theory` 关注对称 / 非对称、per-tensor / per-channel、误差和 scale。
- `PTQ / QAT` 关注量化发生在训练前还是训练中。
- `GPTQ / AWQ` 关注训练后量化时怎么尽量保精度。
- `FP8` 关注新硬件和训练 / 推理中的低精度表示。
- `KV Cache Quantization` 关注推理侧缓存压缩。
- `Deployment Choice` 关注在精度、显存、带宽、吞吐和成本之间怎么做决策。

## Part 1 相关前置

- [1A](../../01_Hardware_Math_and_Systems/1A.md)：先看数值精度、scale 和误差直觉，知道量化到底在压什么。
- [1B](../../01_Hardware_Math_and_Systems/1B.md)：先看单卡硬件、TensorCore 和访存直觉，知道量化为什么会同时影响显存、带宽和吞吐。

## 量化链路总图

```text
floating-point model
  │
  ├── PTQ ──校准/重参数化──► weight-only / activation quantization
  │
  ├── QAT ──训练中模拟量化误差──► low-bit aware model
  │
  ├── GPTQ / AWQ ──后训练压缩──► high-accuracy low-bit weights
  │
  ├── FP8 ──硬件友好低精度路径──► training / inference low precision
  │
  └── KV cache quant ──缓存压缩──► lower inference memory
            │
            ▼
 deployment report ── memory / latency / throughput / quality / cost
```

## Task1-6 主线

| Task | 主题 | 推荐小节 | 学完应能回答 |
|:---|:---|:---|:---|
| Task1 | 量化基础与硬件直觉 | Part01 [01 Data Types and Precision](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)、[12 TensorCore and Mixed Precision](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)、[21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) | 量化为什么能同时影响显存、带宽和吞吐？ |
| Task2 | PTQ / QAT 的训练时机 | Part01 [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) + Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | 什么时候先做 PTQ，什么时候应该考虑 QAT 或 LoRA/QLoRA？ |
| Task3 | GPTQ / AWQ 的后训练压缩 | Part01 [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) + Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[40 GPTQ and AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) | 训练后量化如何尽量保住精度？ |
| Task4 | FP8 与 KV Cache Quantization | Part01 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)、[12 TensorCore and Mixed Precision](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) + Part02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | FP8 和 KV cache quant 改的是哪一类成本？ |
| Task5 | 量化部署与服务选型 | Part02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) | 量化如何和部署、调度、显存预算一起决策？ |
| Task6 | 量化项目收口 | Part02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) | 如何用统一 workload 判断量化方案是否值得切换？ |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `21` | 量化理论、误差、scale、PTQ / QAT / GPTQ / AWQ 基础 | [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) |
| `25` | W8A16 权重量化与最小实现 | [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) |
| `26` | QLoRA 与 4-bit 量化 | [26 QLoRA and 4bit Quantization](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) |
| `67` | 量化推理与部署 | [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) |
| `40` | GPTQ / AWQ 权重量化 | [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) |
| `41` | FP8 与 KV Cache 量化 | [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) |
| `10` | Triton 量化算子实现 | [10 Triton Quantization](../../03_Triton_Kernels/10_Triton_Quantization.ipynb) |

## 推荐入口

- 如果你从零学量化，先看 Task1-3。
- 如果你关心训练时机和微调关系，先看 Task2，再回到 `26`。
- 如果你关心部署和服务选型，先看 Task4-6。
- 如果你关心 kernel 实现，再跳到 Part 03 的 `10`。

## 正文页

- [量化与压缩正文](./casebook.md)：指标口径、PTQ/QAT/GPTQ/AWQ/FP8 对照、常见误区和部署清单。
- [量化与压缩深入阅读](./walkthrough.md)：从一个模型压缩决策开始，连续走到量化部署报告。

## 相关专题

- [推理优化专题](../inference_optimization/intro.md)：当量化主要服务于吞吐、TTFT、TPOT 和 cache 预算时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当量化主要服务于 VRAM 压缩和性能调优时看这里。
- [训练微调闭环专题](../fine_tuning_training/intro.md)：当量化和 LoRA / QLoRA / 训练时机绑在一起时看这里。
- [算子优化与 Kernel 实战专题](../compiler_graph_optimization/intro.md)：当你要把量化落到 Triton / kernel 视角时看这里。

## 专题状态

当前为专题入口页。后续优先补 `casebook.md` 和 `walkthrough.md`，再把 `21 / 25 / 26 / 40 / 41 / 67 / 10` 的阅读导流整理得更紧。
