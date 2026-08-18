# 显存优化专题

## 专题定位

本专题用于串起显存优化主线：先看训练侧为什么会 OOM，再看推理侧为什么会被 KV cache 顶住预算，最后把 checkpointing、offload、量化和 benchmark 一起收回端到端 trade-off 结论。这里重点关注 `peak memory / VRAM ledger / trade-off`；如果问题先表现为请求链路变慢，应优先转到推理优化专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part00 / Part01 / Part02` 的具体小节；最后一列的 `01-06` 是专题正文页，只负责解释和串联。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | 显存与性能认知底座 | `P1:03 -> 06 -> 13` | [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md) |
| Task2 | 训练侧显存优化 | `12 -> 19 -> 42` | [02 Training Memory Pressure](./02_training_memory_pressure.md) |
| Task3 | 训练侧验证与调优 | `73 -> 74 -> 75 -> 76` | [03 Checkpointing and Offload](./03_checkpointing_and_offload.md) |
| Task4 | 推理侧显存优化 | `P1:11 -> 22 -> 24 -> 34 -> 37` | [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md) |
| Task5 | 量化作为显存手段 | `P1:21 -> 25 -> 26 -> 40 -> 41 -> 67` | [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md) |
| Task6 | benchmark 与 trade-off 收口 | `74 -> 75 -> 76 -> 67` | [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md) |

## 正文与跳转

先按上面的 `Task1-6` 走 notebook 主线；遇到“训练侧和推理侧的预算问题怎么区分”“为什么峰值降了但系统未必更好”时，再回来看对应的专题正文 `01-06`。想看汇总版就进 [显存优化与性能调优正文](./casebook.md)，想按完整故事线走一遍就进 [显存优化与性能调优深入阅读](./walkthrough.md)。

如果问题已经跨到别的专题：
[反向传播与训练机制专题](../backpropagation_training_mechanism/intro.md) 负责训练侧 activation 与 backward 基础，[推理优化专题](../inference_optimization/intro.md) 负责请求链路速度问题，[量化与压缩专题](../quantization/intro.md) 负责低比特压缩路线，[Profiling 专题](../profiling/intro.md) 负责证据链和瓶颈定位，[通信与并行专题](../communication_parallel/intro.md) 负责多卡切分与参数分摊边界。
