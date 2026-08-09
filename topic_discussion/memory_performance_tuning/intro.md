# 显存优化与性能调优专题

## 专题概览
本专题用于沉淀 VRAM、activation、checkpointing、offload、KV cache 和 benchmark 相关内容，回答“怎么处理显存压力并做端到端性能调优”。
`Part 0E` 也是这个专题的前置桥，因为它已经把显存观察、调试和性能判断串在了一起。

## 职责边界

这个专题负责显存压力、缓存增长和端到端性能调优，不负责并行策略本身，也不负责编译链路本体。

- `VRAM / Memory Ledger` 关注内存账本、峰值占用和资源分配。
- `Activation / Checkpointing / Offload` 关注训练侧显存压力和时间换空间。
- `KV Cache` 关注推理侧缓存增长、布局和复用。
- `Benchmark / Profiling` 关注把性能问题量化成可比较指标。
- `Deployment Tuning` 关注量化、推理和部署场景中的显存/性能权衡。

## 三条显存线

显存优化可以按三条主线来读。它们共享 profiling 和 benchmark，但关注点不同。

| 主线 | 核心问题 | 代表内容 |
|:---|:---|:---|
| 训练显存 | 训练时为什么会 OOM，怎么用时间换空间 | `12`, `19`, `42`, `73`, `74` |
| 推理显存 | KV cache 为什么涨，怎么把缓存压进预算 | `22`, `24`, `34`, `36`, `37`, `41` |
| 量化显存 | 如何通过压缩权重 / cache 降低显存占用 | `25`, `40`, `41`, `67` |

和 `推理优化专题` 的区别是：

- 本专题先看资源为什么超预算、怎么压进预算，重点是 `peak memory / VRAM ledger / 资源 trade-off`。
- 推理专题先看请求怎么更快出 token，重点是 `TTFT / TPOT / throughput`。
- 两边都会碰到 `KV cache`、量化和 benchmark，但本专题把它们当成资源约束和调优成本来看。

## 与推理优化专题的交叉路由

这两个专题会共享一些 notebook，但不共享同一个问题目标。最容易重合的是 `KV cache`、量化、benchmark 和调度。

| 重合内容 | 本专题先回答什么 | 推理专题先回答什么 |
|:---|:---|:---|
| `KV cache` | 为什么把 `peak memory` 顶高、预算怎么估、分页/压缩后能省多少 | 为什么拖慢 `TTFT / TPOT / throughput`，如何通过 reuse、paging、scheduling 让请求更快 |
| `量化` | 权重、KV cache 和激活压缩后能省多少 VRAM，代价是什么 | 是否让部署链路更快，是否值得换精度或后端 |
| `benchmark / profiling` | 哪个资源对象最占显存，时间换空间是否划算 | 哪段请求链路最慢，优化是否真让延迟和吞吐改善 |
| `调度` | 资源排布是否把 cache、buffer、activation 顶到预算外 | batch、prefill、decode 如何排队才更快 |

实用判断可以保持简单：

- 如果你在问“为什么装不下”，先看本专题。
- 如果你在问“为什么慢”，先看推理专题。
- 如果你在问“既慢又占显存”，优先检查 `KV cache`、量化和调度这三条交叉线。

## 对应来源

| 来源 | 适合纳入的内容 |
|:---|:---|
| `Part 0E` | 调试、显存和性能判断的前置桥 |
| `Part 1` | VRAM 估算、memory ledger、profiling 基础 |
| `Part 2.5` | 反向传播、gradient accumulation、activation checkpointing、offload |
| `Part 2.6` | FlashAttention、KV cache、推理侧显存观察 |
| `Part 2.9` | 训练/推理性能分析、量化部署与 benchmark 闭环 |

## 章节跳转

| 章节 | 你会看到什么 | 跳转 |
|:---|:---|:---|
| `Part 1B` | 单卡硬件、访存和显存估算的基础入口 | [1B 单卡硬件与访存优化](../01_Hardware_Math_and_Systems/1B.md) |
| `0E` | 调试与性能前置桥，先把显存与性能判断习惯立住 | [0E 调试与性能](../../00_Prerequisites/0E.md) |
| `0E-17` | profiling 的基础入口和瓶颈定位 | [17 PyTorch Profiling Basics](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb) |
| `0E-18` | 显存账本与优化手段 | [18 Memory Profiling and Optimization](../../00_Prerequisites/18_Memory_Profiling_and_Optimization.ipynb) |
| `0E-19` | 最小排错和异常定位 | [19 Debugging and Anomaly Localization](../../00_Prerequisites/19_Debugging_and_Anomaly_Localization.ipynb) |
| `0E-20` | 性能判断和优化决策 | [20 Profiling and Memory Ledger](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb) |
| `06` | VRAM 计算与 ZeRO 的显存收益 | [06 VRAM Calculation and ZeRO](../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb) |
| `13` | profiling 与瓶颈定位的方法入口 | [13 Profiling and Bottleneck Analysis](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb) |
| `2.5` | 反向传播与显存优化主线 | [2.5 反向传播与显存优化](../02_PyTorch_Algorithms/2_5.md) |
| `12` | 梯度累积与有效 batch 的显存控制 | [12 Gradient Accumulation](../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb) |
| `19` | checkpointing 的显存 trade-off | [19 Activation Checkpointing and Activation Offload](../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb) |
| `42` | activation offload 的搬运路线 | [42 Activation Offload](../02_PyTorch_Algorithms/42_Activation_Offload.ipynb) |
| `2.6` | 推理侧缓存和显存路径 | [2.6 核心推理优化](../02_PyTorch_Algorithms/2_6.md) |
| `22` | PagedAttention 的 KV cache 管理 | [22 vLLM PagedAttention](../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb) |
| `73` | 训练性能分析与显存对比项目 | [73 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb) |
| `74` | profiling 驱动的端到端优化项目 | [74 Profiling Driven End-to-End Optimization](../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb) |
| `67` | 量化推理与部署中的显存权衡 | [67 Quantized Inference and Deployment](../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) |

## 推荐入口

### Task1：显存与性能认知底座
- `Part 1B / 06 / 13`
- 前置桥：`0E -> 17 -> 18 -> 19 -> 20`

目标：先立显存账本和 profiling 口径，知道峰值、带宽、重算和搬运分别在看什么。

### Task2：训练侧显存优化
- `12 -> 19 -> 42`

目标：把 `effective batch`、`activation`、`checkpointing` 和 `offload` 放到同一条链上看。

### Task3：训练侧性能验证
- `73 -> 74`

目标：确认优化前后到底是省了显存，还是把时间赔掉了。

### Task4：推理侧显存优化
- `2.6 -> 22 -> 24 -> 34 -> 36 -> 37 -> 41`

目标：看清 KV cache、prefix 复用、分页和调度如何一起影响显存。

### Task5：量化作为显存手段
- `25 -> 40 -> 41 -> 67`

目标：把量化当成显存调优工具，评估权重、cache 和部署收益。

### Task6：项目收口
- `73 -> 74 -> 67`

目标：把训练、推理和量化的优化结果收进 benchmark report。

## 入口摘要

- 第一入口：`Part 1B` + `06 -> 13`，先把显存账本、VRAM 计算和瓶颈定位立住。
- 第二入口：`12 -> 19 -> 42 -> 73` / `2.6 -> 22 -> 24 -> 34 -> 36 -> 37 -> 41`，把训练侧和推理侧的显存压力看清楚。
- 验证入口：`74 -> 67`，把 profiling 驱动的优化和量化部署的收益验证收进闭环。

## 正文页

- [01 VRAM Ledger and Metrics](./01_vram_ledger_and_metrics.md)
- [02 Training Memory Pressure](./02_training_memory_pressure.md)
- [03 Checkpointing and Offload](./03_checkpointing_and_offload.md)
- [04 Inference Cache and Memory Budget](./04_inference_cache_and_memory_budget.md)
- [05 Quantization as a Memory Tool](./05_quantization_as_a_memory_tool.md)
- [06 Benchmark and Trade-off Decision](./06_benchmark_and_tradeoff_decision.md)
- [07 Visual Assets](./07_visual_assets.md)
- [显存优化与性能调优正文](./casebook.md)：按“训练侧 / 推理侧 / 验证侧”展开正文，适合做更细的显存案例和调优记录。
- [显存优化与性能调优深入阅读](./walkthrough.md)：按完整调优故事展开，适合想看连续推演的人。

## 相关专题

- [Profiling 专题](../profiling/intro.md)：当你需要先把瓶颈、热点和收益先量化出来时先看这里。
- [推理优化专题](../inference_optimization/intro.md)：当显存压力主要来自推理链路里的 cache、prefill 或 decode 时先看这里。
- [量化与压缩专题](../quantization/intro.md)：当显存压力主要来自模型体积、KV cache 或部署侧压缩时先看这里。
- [通信与并行专题](../communication_parallel/intro.md)：当显存压力和多卡切分、参数分摊一起出现时先看这里。

## Part 1 / Part 2 入口顺序

### Part 1 入口

- 先看 `Part 1B`，把单卡硬件、访存和显存估算的基础账本立住。
- 再看 `06 -> 13`，把 VRAM 计算、ZeRO 收益和瓶颈定位先串起来。
- 如果想补前置桥，再从 `0E -> 17 -> 18 -> 19 -> 20` 过一遍。

### Part 2 入口

- 先看 `12 -> 19 -> 42 -> 73`，把训练侧 batch、activation、checkpointing、offload 和性能分析串起来。
- 再看 `2.6 -> 22 -> 24 -> 34 -> 36 -> 37 -> 41`，把推理侧 KV cache、前缀复用和调度路径串起来。
- 最后看 `74`，把 profiling 驱动的端到端优化补成闭环。

## 读法建议

- 如果你还没看 `0E`，建议先补它，再进这个专题。
- 如果你想先补前置桥，可以按 `0E -> 17 -> 18 -> 19 -> 20` 这条线过一遍，先把显存账本、排错习惯和性能判断立住。
- 如果你关心“训练显存为什么爆”，先看 `12 -> 19 -> 42 -> 73`。
- 如果你关心“推理显存为什么涨”，先看 `2.6 -> 22 -> 24 -> 34 -> 36 -> 37 -> 41`。
- 如果你关心“怎么证明优化有效”，先看 `13 -> 74`。

## 建设方式

- 先把入口和路径讲清楚，再把正文页里的资源对象、案例和检查清单补深。
- 新增内容优先回收到 `2.5 / 2.6 / 73 / 74 / 67` 这几条线。
- 导读页只负责告诉读者“从哪进”，不再重复正文里的判断框架。

## 专题状态
本专题已更新为 `01-06 + 07_visual_assets` 的解释层结构。它的作用是把训练显存、推理缓存、量化预算和 benchmark 收束成一套显存判断框架，而不是复述 Part02 目录。
