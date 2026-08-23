# 显存优化专题

## 页面导语

本专题围绕训练与推理显存账本、预算压力和资源取舍，最终形成可验证的显存优化决策。

## 如何开始

推荐从 Part 02 的 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 开始；如果还不熟悉显存和性能指标，先回补 Part 01 的 GPU 与显存基础。

- 必读前置：先完成 [17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)，再补 [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb)。
- 按需回补：需要看激活与 loss 反传时阅读扩展 [18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb)；不熟悉显存指标时先补 Part 01 的 GPU 与显存基础。
- 项目入口：完成 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 后，进入 [73](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) → [74](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) 的项目决策链。

## 主学习线与分级

`Task0-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task0 | 自动微分基础 | [17 Autograd Basics](../../02_PyTorch_Algorithms/17_Autograd_Basics.ipynb)；扩展：[18 Activation and Loss Backward](../../02_PyTorch_Algorithms/18_Activation_and_Loss_Backward.ipynb) | — |
| Task1 | 显存与性能认知底座 | [Part 01:03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) | [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md) |
| Task2 | 训练侧显存优化 | [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) → [19 Activation Checkpointing](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)；扩展：[42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) | [02 Training Memory Pressure](./02_training_memory_pressure.md) |
| Task3 | 训练侧验证与调优 | [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76 Activation / Checkpoint / Offload](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)；扩展：[75 Memory Budget Compression](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) | [03 Checkpointing and Offload](./03_checkpointing_and_offload.md) |
| Task4 | 推理侧显存优化 | 核心：[11 KV Cache Growth](../../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.ipynb) → [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb) → [66 Inference Performance](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)；扩展：[24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)、[37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) | [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md) |
| Task5 | 量化作为显存手段 | 核心：[21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)；扩展：[40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) | [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md) |
| Task6 | Profiling 驱动的显存优化最终收口 | [43 Unified Memory Management](../../02_PyTorch_Algorithms/43_Unified_Memory_Management.ipynb) → [44 Auto Tuning Framework](../../02_PyTorch_Algorithms/44_Auto_Tuning_Framework.ipynb) → [45 Memory Cut Planning](../../02_PyTorch_Algorithms/45_Memory_Cut_Planning.ipynb) → [74 Profiling Driven Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) | [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md) |

### Task1 的共享基础与本路线阅读视角

[03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) 和 [20 FlashAttention Sim](../../02_PyTorch_Algorithms/20_FlashAttention_Sim.ipynb) 是多个学习路线共享的基础 Notebook。本路线不要求把它们重新学习成“显存专属内容”，而是带着下面的问题阅读：

- GPU 存储层级的容量与带宽，如何限制训练规模？
- 参数、梯度、优化器状态、激活和 Attention 临时空间分别是什么？
- 哪些问题属于容量不足，哪些问题属于带宽或 IO 压力？
- FlashAttention 减少的是哪类临时空间，和 checkpoint / offload 有什么不同？

同一内容在推理优化路线中会进一步连接 KV Cache、Prefill/Decode 和请求并发；在算子与编译优化路线中会进一步连接 Tiling、数据复用、算子融合和 Kernel 执行效率。这里先保留共同机制，只切换观察角度。

### Task1 输出

完成一张简化显存账本，并对一个训练 workload 做初步判断：

1. 当前主要瓶颈是容量、带宽、临时空间，还是状态驻留？
2. 如果激活是主要瓶颈，下一步应进入 Gradient Accumulation、Checkpointing 还是 Offload？
3. 哪些结论还只是机制推断，必须交给 73 / 76 的真实 GPU 测量？

### 核心与扩展分级

Task 3、Task 4 和 Task 5 都采用“核心路径 + 扩展路径”，避免把高压力 workload、真实 backend、特定推理引擎和高级量化工具变成所有学习者的硬性前置。

| Task | 核心路径 | 扩展路径 | 环境级别 |
|:---|:---|:---|:---|
| Task3 训练侧显存 | [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) 建立基线 → [76 Activation / Checkpoint / Offload](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) checkpoint 对比 → [75 Memory Budget Compression](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) 预算决策 | [76](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) offload / hybrid / pressure workload → [75](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) 严格预算筛选 | 核心 Practice-P1 单 GPU；扩展 Practice-P1 高压力实验 |
| Task4 推理侧显存 | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)，再做 [66 Inference Performance](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 单 backend 最小验证 | [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb) → [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) → [66](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 并发 / 多 backend 对比 | 机制 Practice-P0/P1；`66` smoke test Practice-P2；扩展 Practice-P2 |
| Task5 量化显存 | [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 本地量化模型加载 | [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) → [41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) → [67](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) backend 部署 | 核心 Practice-P0/P1；扩展 Practice-P2 |

核心路径的目标是理解机制并完成一次可复现实验：Task 3 记录 step time、吞吐、峰值显存和 loss，Task 4 记录 KV Cache、延迟、吞吐和峰值显存，Task 5 记录模型占用、峰值显存、速度和质量。扩展路径再测试 offload / hybrid、高压力序列长度、vLLM / SGLang、并发、长上下文或具体量化 backend。

`66` 的机制学习属于核心路径，但只要真正启动 vLLM / SGLang 就进入 Practice-P2；没有 backend 时可完成机制和模拟验证。`67` 的本地模型加载属于 Practice-P1，真实 vLLM / SGLang 量化部署属于 Practice-P2。并发压测和多方案比较属于扩展路径。`26 QLoRA` 继续服务训练微调和训练侧显存路线，不并入 Task 5 的推理量化主线。

## 高级路线：分布式显存与系统级预算

上面的 `Task0-6` 是单机显存优化主线，默认先理解单 GPU 上的容量、带宽、激活、KV Cache 和量化问题。完成主线后，如果需要继续研究多卡训练、状态分摊和通信代价，再进入下面的高级路线；它不是单机主线的硬性前置。

```text
数据类型与参数规模
  → GPU 与 Attention 显存
  → 通信拓扑与训练状态账本
  → ZeRO / 并行策略
  → 异构调度与分布式项目
```

| 阶段 | 学习内容 | 入口 | 目标 |
|:---|:---|:---|:---|
| A0 | 精度、参数与规模估算 | [01 数据类型与精度](../../01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.ipynb) → [02 参数量与 FLOPs](../../01_Hardware_Math_and_Systems/02_LLM_Params_and_FLOPs.ipynb) | 建立参数、dtype 和预算的换算关系 |
| A1 | 单卡硬件与 Attention 显存 | [03 GPU Architecture and Memory](../../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.ipynb) → [04 Attention Memory Optimization](../../01_Hardware_Math_and_Systems/04_Attention_Memory_Optimization.ipynb) | 区分容量、带宽、临时空间和 KV Cache |
| A2 | 通信与训练状态分摊 | [05 Communication Topologies](../../01_Hardware_Math_and_Systems/05_Communication_Topologies.ipynb) → [06 VRAM Calculation and ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) | 理解 DDP、ZeRO 和通信代价 |
| A3 | 异构调度与并行决策 | [07 CPU/GPU Heterogeneous Scheduling](../../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.ipynb) → [26 Parallel Strategy Decision](../../01_Hardware_Math_and_Systems/26_Parallel_Strategy_Decision_Framework.ipynb) → [27 Communication Scheduling](../../01_Hardware_Math_and_Systems/27_Communication_Scheduling_Optimization.ipynb) | 判断状态放置、切分和通信时机 |
| A4 | 分布式项目验证 | [79 Distributed Parallel Benchmark](../../02_PyTorch_Algorithms/79_Distributed_Parallel_Benchmark.ipynb) → [80 MoE Expert Parallel Benchmark](../../02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb) → [81 Distributed Inference Project](../../02_PyTorch_Algorithms/81_Distributed_Inference_Project.ipynb) | 在真实或模拟多卡环境中验证预算与通信权衡 |

高级路线的核心问题是“状态如何在多卡之间分摊，以及显存下降是否换来了通信和调度代价”。`06` 负责理论账本，`79-81` 负责项目验证；不要把 ZeRO 的理论上限直接当成真实可训练规模。

## 学习方式与项目产出

先按上面的 `Task0-6` 走 Notebook 主线；核心路径用于建立显存对象、生命周期和优化手段，扩展路径用于高压力 workload、真实 backend 或更严格预算。需要连续理解概念时阅读专题正文 `01-06`，需要判断表和项目分流时阅读[显存优化与性能调优正文](./casebook.md)，需要完整串联路线时阅读[显存优化与性能调优深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责训练侧 activation 与 backward 基础，[推理优化专题](../inference_optimization/intro.md) 负责请求链路速度问题，[量化与压缩专题](../quantization/intro.md) 负责低比特压缩路线，[Profiling 专题](../profiling/intro.md) 负责证据链和瓶颈定位，[通信与并行专题](../communication_parallel/intro.md) 负责多卡切分与参数分摊边界。

### 项目产出

推荐顺序为 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76 Activation / Checkpoint / Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) → [74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)。73 建立训练侧测量基线，76 比较训练侧显存策略，75 形成训练侧预算决策，74 使用 profiling 对显存优化方案做端到端最终验证。

## 环境与验证

基础机制可 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON 结果，不能只根据单次峰值变化判定优化成功。

项目运行顺序、GPU 环境检查、结果文件和 74 profiling 要求见[73–76 显存优化项目验证清单](../../docs/verification/memory_projects.md)。
