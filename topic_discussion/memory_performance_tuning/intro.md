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

### 核心与扩展分级

Task 3、Task 4 和 Task 5 都采用“核心路径 + 扩展路径”，避免把高压力 workload、真实 backend、特定推理引擎和高级量化工具变成所有学习者的硬性前置。

| Task | 核心路径 | 扩展路径 | 环境级别 |
|:---|:---|:---|:---|
| Task3 训练侧显存 | [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) 建立基线 → [76 Activation / Checkpoint / Offload](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) checkpoint 对比 → [75 Memory Budget Compression](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) 预算决策 | [76](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) offload / hybrid / pressure workload → [75](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) 严格预算筛选 | 核心 Practice-P1 单 GPU；扩展 Practice-P1 高压力实验 |
| Task4 推理侧显存 | [22 PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) → [34 Prefix Caching](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)，再做 [66 Inference Performance](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 单 backend 最小验证 | [24 RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb) → [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb) → [66](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 并发 / 多 backend 对比 | 机制 Practice-P0/P1；`66` smoke test Practice-P2；扩展 Practice-P2 |
| Task5 量化显存 | [21 Quantization Theory](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb) → [25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) → [67 Quantized Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 本地量化模型加载 | [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) → [41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) → [67](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) backend 部署 | 核心 Practice-P0/P1；扩展 Practice-P2 |

核心路径的目标是理解机制并完成一次可复现实验：Task 3 记录 step time、吞吐、峰值显存和 loss，Task 4 记录 KV Cache、延迟、吞吐和峰值显存，Task 5 记录模型占用、峰值显存、速度和质量。扩展路径再测试 offload / hybrid、高压力序列长度、vLLM / SGLang、并发、长上下文或具体量化 backend。

`66` 的机制学习属于核心路径，但只要真正启动 vLLM / SGLang 就进入 Practice-P2；没有 backend 时可完成机制和模拟验证。`67` 的本地模型加载属于 Practice-P1，真实 vLLM / SGLang 量化部署属于 Practice-P2。并发压测和多方案比较属于扩展路径。`26 QLoRA` 继续服务训练微调和训练侧显存路线，不并入 Task 5 的推理量化主线。

## 学习方式与项目产出

先按上面的 `Task0-6` 走 Notebook 主线；核心路径用于建立显存对象、生命周期和优化手段，扩展路径用于高压力 workload、真实 backend 或更严格预算。需要连续理解概念时阅读专题正文 `01-06`，需要判断表和项目分流时阅读[显存优化与性能调优正文](./casebook.md)，需要完整串联路线时阅读[显存优化与性能调优深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责训练侧 activation 与 backward 基础，[推理优化专题](../inference_optimization/intro.md) 负责请求链路速度问题，[量化与压缩专题](../quantization/intro.md) 负责低比特压缩路线，[Profiling 专题](../profiling/intro.md) 负责证据链和瓶颈定位，[通信与并行专题](../communication_parallel/intro.md) 负责多卡切分与参数分摊边界。

### 项目产出

推荐顺序为 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76 Activation / Checkpoint / Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb) → [74 Profiling 驱动的端到端优化](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)。73 建立训练侧测量基线，76 比较训练侧显存策略，75 形成训练侧预算决策，74 使用 profiling 对显存优化方案做端到端最终验证。

## 环境与验证

基础机制可 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON 结果，不能只根据单次峰值变化判定优化成功。

项目运行顺序、GPU 环境检查、结果文件和 74 profiling 要求见[73–76 显存优化项目验证清单](../../docs/verification/memory_projects.md)。
