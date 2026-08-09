# 量化与压缩专题

## 专题概览

本专题不是从零发明一条新路线，而是**承接 `Part00-02` 已经存在的学习路径**，把分散在不同 Part 里的量化内容重新组织成一条“低比特表示如何进入系统”的故事线。它主要回答三个问题：

- 量化到底在压什么，压的是权重、激活，还是 KV cache？
- PTQ、QAT、GPTQ、AWQ、FP8 分别在什么环节起作用？
- 怎么把量化结果放回推理、显存和部署约束里做选型？

这个专题不是推理优化的附属页，也不是 Part01/02 的文件目录副本。它是一个**方法轴**：

- `Part00` 提供数值、误差和调试直觉；
- `Part01` 提供精度、硬件、访存和低比特执行背景；
- `Part02` 提供量化实现、量化部署和项目验证；
- 横向专题负责把这些内容重组为“对象 -> 时机 -> 适配 -> 压缩 -> 执行栈 -> 决策”的逻辑链。

## 职责边界

这个专题负责量化理论、低比特表示、量化训练和量化部署选型，不负责 FlashAttention、decode scheduling 或一般性的 serving 调度。

- `Quantization Theory` 关注对称 / 非对称、per-tensor / per-channel、误差和 scale。
- `PTQ / QAT` 关注量化发生在训练前还是训练中。
- `GPTQ / AWQ` 关注训练后量化时怎么尽量保精度。
- `FP8` 关注新硬件和训练 / 推理中的低精度表示。
- `KV Cache Quantization` 关注推理侧缓存压缩。
- `Deployment Choice` 关注在精度、显存、带宽、吞吐和成本之间怎么做决策。

## 承接已有学习路线

这个专题的正文应当建立在已有主线上，而不是脱离主线重新开一门课。

### Part00：数值与调试前置

- 数值表示、误差传播和调试习惯，负责建立“为什么低比特会出问题”的直觉。

### Part01：硬件与精度前置

- [1A](../../01_Hardware_Math_and_Systems/1A.md)：先看数值精度、scale 和误差直觉，知道量化到底在压什么。
- [1B](../../01_Hardware_Math_and_Systems/1B.md)：先看单卡硬件、TensorCore 和访存直觉，知道量化为什么会同时影响显存、带宽和吞吐。

### Part02：实现、部署与项目收口

- `25 / 26 / 40 / 41 / 67` 负责把量化放到实现、微调、推理缓存和部署验证里。
- `66` 负责把量化收益放回统一 workload 做最后决策。

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

`Task1-6` 仍然保留，因为它们承接的是已有 notebook 学习顺序；但它们只是学习内容路径，不是专题本体。

| Task | 主题 | 推荐小节 | 学完应能回答 |
|:---|:---|:---|:---|
| Task1 | 量化基础与硬件直觉 | Part01 [01 Data Types and Precision](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb)、[12 TensorCore and Mixed Precision](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb)、[21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) | 量化为什么能同时影响显存、带宽和吞吐？ |
| Task2 | PTQ / QAT 的训练时机 | Part01 [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) + Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[26 QLoRA](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.ipynb) | 什么时候先做 PTQ，什么时候应该考虑 QAT 或 LoRA/QLoRA？ |
| Task3 | GPTQ / AWQ 的后训练压缩 | Part01 [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) + Part02 [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb)、[40 GPTQ and AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) | 训练后量化如何尽量保住精度？ |
| Task4 | FP8 与 KV Cache Quantization | Part01 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)、[12 TensorCore and Mixed Precision](../../01_Hardware_Math_and_Systems/12_TensorCore_and_Mixed_Precision.ipynb) + Part02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | FP8 和 KV cache quant 改的是哪一类成本？ |
| Task5 | 量化部署与服务选型 | Part02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) | 量化如何和部署、调度、显存预算一起决策？ |
| Task6 | 量化项目收口 | Part02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) | 如何用统一 workload 判断量化方案是否值得切换？ |

## 为什么这个专题不能退化成索引

如果这里只是把 `21 / 25 / 26 / 40 / 41 / 67 / 10` 列出来，它就仍然只是“去哪看”的答案。横向专题真正应该补的是三层厚度：

- 文字串联：为什么模型会走向低比特表示，这条路线内部的核心矛盾是什么。
- 文献锚点：这些方法是谁提出的、分别为了解决哪类误差或部署问题。
- 可视化：让读者看到图就知道当前在压哪一类对象、下一步该切哪条路线。

因此，下面的 `01-06` 不是文件索引，而是把主线重组成一条可独立阅读的故事线。

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

## 01-06 骨架

这 6 个小节是知识组织层，不要求和 `Task1-6` 一一对应。`Task1-6` 负责学习内容顺序，`01-06` 负责把已有路线里的量化内容重组为一条完整故事线：

- 先分清压缩对象与误差；
- 再决定量化什么时候介入；
- 再看是否需要训练适配；
- 再看后训练压缩和权重量化；
- 再看执行栈和缓存预算；
- 最后回到部署和 benchmark。

| 章节 | 你会得到什么 | 适合先从哪里进入 |
|:---|:---|:---|
| `01` | 量化对象、误差和粒度直觉 | 先分不清压缩对象时 |
| `02` | PTQ / QAT 的介入时机 | 先判断要不要回到训练时 |
| `03` | 低比特训练适配 | PTQ 不够、但还有训练预算时 |
| `04` | 权重量化与后训练压缩 | 权重驻留是首要预算问题时 |
| `05` | FP8 与 KV cache quant | 问题更像执行栈或缓存预算时 |
| `06` | 部署与 benchmark 决策 | 已经有候选方案，要判断值不值得切换时 |

## 正文页

- [01 Quantization Object and Error](./01_quantization_object_and_error.md)
- [02 PTQ and QAT Timing](./02_ptq_and_qat_timing.md)
- [03 Low-Bit Training Adaptation](./03_low_bit_training_adaptation.md)
- [04 Weight-Only Compression](./04_weight_only_compression.md)
- [05 FP8 and KV Cache Quantization](./05_fp8_and_kv_cache_quantization.md)
- [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md)
- [07 Visual Assets](./07_visual_assets.md)
- [量化与压缩正文](./casebook.md)：指标口径、PTQ/QAT/GPTQ/AWQ/FP8 对照、常见误区和部署清单。
- [量化与压缩深入阅读](./walkthrough.md)：从一个模型压缩决策开始，连续走到量化部署报告。

## 相关专题

- [推理优化专题](../inference_optimization/intro.md)：当量化主要服务于吞吐、TTFT、TPOT 和 cache 预算时看这里。
- [显存优化与性能调优专题](../memory_performance_tuning/intro.md)：当量化主要服务于 VRAM 压缩和性能调优时看这里。
- [训练微调闭环专题](../fine_tuning_training/intro.md)：当量化和 LoRA / QLoRA / 训练时机绑在一起时看这里。
- [算子优化与 Kernel 实战专题](../compiler_graph_optimization/intro.md)：当你要把量化落到 Triton / kernel 视角时看这里。

## 专题状态

本专题已更新为 `01-06 + 07_visual_assets` 的解释层结构。当前已完成正文层，下一步优先补图册与更细的论文锚点。
