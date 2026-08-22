# 显存优化专题

## 专题定位

本专题用于串起显存优化主线：先看训练侧为什么会 OOM，再看推理侧为什么会被 KV cache 顶住预算，最后把 checkpointing、offload、量化和 benchmark 一起收回端到端 trade-off 结论。这里重点关注 `peak memory / VRAM ledger / trade-off`；如果问题先表现为请求链路变慢，应优先转到推理优化专题。

## Infra 层定位

显存优化主要位于 `L1 硬件与内存层`、`L2 系统软件与访存层`、`L3 框架与运行时`，并延伸到 L4 的 KV Cache、量化和 Serving 调度。每项策略都要同时检查容量、带宽、计算重算、CPU-GPU 搬运和端到端性能，不能只比较峰值显存。

## 同一内容的显存目标

本专题把 checkpointing、offload、paging、KV Cache 和量化看成“资源预算策略”：核心问题是减少哪类状态的驻留，以及代价转移到了重算、带宽、通信还是质量。量化在这里首先是显存工具，要先回答模型是否装得下、上下文或并发是否能提高，再用实际 workload 验证速度和质量。

与推理专题共享同一来源 Notebook 时，所有学习者先理解共同机制，再按目标选择指标：显存目标关注状态账本、peak memory、带宽、并发容量和 OOM 边界，最终输出是显存上限、吞吐下限、质量下限下的 `accept / tune / reject` 决策。

## 推荐入口

推荐从 Part 02 的 [2.5 反向传播与显存优化](../../02_PyTorch_Algorithms/2_5.md) 开始；如果还不熟悉显存和性能指标，先回补 Part 01 的 GPU 与显存基础。

## 前置阅读

- Part 00：Autograd、反向传播和训练循环基础。
- Part 01：GPU 架构、显存账本和 profiling 基础。
- Part 02：优先完成 2.5，再进入 73、76 和 75 的项目决策链。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 00 / Part 01 / Part 02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 显存与性能认知底座 | `Part 01:03 -> 06 -> 13` | [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md) |
| Task2 | 训练侧显存优化 | `12 -> 19 -> 42` | [02 Training Memory Pressure](./02_training_memory_pressure.md) |
| Task3 | 训练侧验证与调优 | `73 -> 74 -> 75 -> 76` | [03 Checkpointing and Offload](./03_checkpointing_and_offload.md) |
| Task4 | 推理侧显存优化 | `Part 01:11 -> 22 -> 24 -> 34 -> 37` | [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md) |
| Task5 | 量化作为显存手段 | `Part 01:21 -> 25 -> 26 -> 40 -> 41 -> 67` | [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md) |
| Task6 | 显存管理、自动调优与 trade-off 收口 | `43 -> 44 -> 45 -> 74 -> 75 -> 76 -> 67` | [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“训练侧和推理侧的预算问题怎么区分”“为什么峰值降了但系统未必更好”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [显存优化与性能调优正文](./casebook.md)，想按完整故事线走一遍就进 [显存优化与性能调优深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责训练侧 activation 与 backward 基础，[推理优化专题](../inference_optimization/intro.md) 负责请求链路速度问题，[量化与压缩专题](../quantization/intro.md) 负责低比特压缩路线，[Profiling 专题](../profiling/intro.md) 负责证据链和瓶颈定位，[通信与并行专题](../communication_parallel/intro.md) 负责多卡切分与参数分摊边界。

## 项目结论

推荐顺序为 [73 训练性能分析](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) → [76 Activation / Checkpoint / Offload 对比](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb) → [75 显存预算压缩](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)。73 负责建立测量基线，76 负责比较具体策略，75 负责形成预算决策。

## 环境与验证

基础机制可 CPU-first；真实训练、显存峰值和策略对比需要 NVIDIA GPU。运行前确认 PyTorch CUDA 可用，并按 Notebook 输出保存 JSON 结果，不能只根据单次峰值变化判定优化成功。
