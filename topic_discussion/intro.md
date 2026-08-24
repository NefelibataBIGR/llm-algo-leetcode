# 专题讨论轴

`topic_discussion` 是跨 `Part 00-04` 的知识组织层。它不替代纵向 notebook 主线，只负责三件事：

- 把 `Part 02` 的主学习路线收成更清楚的入口
- 给跨路线反复出现的方法轴补判断框架
- 给基础机制补解释层，避免读者只记结论、不懂来源

## 主学习路线

如果你还没有明确的性能或系统问题，先从三条主路线中选一条；横向专题用于补方法和机制，不要求全部顺序完成。

| 路线 | 主入口 | 适合什么时候进入 |
|:---|:---|:---|
| 训练微调方向 | [监督微调专题](./fine_tuning_training/intro.md) | 当你要从结构前置、SFT、LoRA 一路走到项目交付时 |
| 推理优化方向 | [推理优化专题](./inference_optimization/intro.md) | 当你要系统理解 prefill、decode、KV cache、服务和 benchmark 时 |
| 显存优化方向 | [显存优化专题](./memory_performance_tuning/intro.md) | 当你要把训练显存、推理 cache、量化预算和 trade-off 串起来时 |

## LLM Infra 五层总览

横向专题统一放回下面这套从下到上的 Infra 结构理解：

| 层级 | 主要内容 | 核心问题 | 边界判断 |
|:---|:---|:---|:---|
| Infra-L1 硬件与基础设施 | GPU/NPU、CPU、HBM、PCIe、NVLink、InfiniBand、SSD | 物理资源提供了什么能力？ | 改的是芯片、容量、带宽、拓扑或物理设备 |
| Infra-L2 系统软件与加速库 | 驱动、CUDA/ROCm、编译器、Triton、NCCL、cuBLAS、FlashAttention | 如何把硬件能力调用出来？ | 改的是 kernel、算子、编译、通信原语或设备运行时 |
| Infra-L3 框架与运行时 | PyTorch、JAX、FSDP、DeepSpeed、Megatron、训练运行时 | 模型计算和状态如何组织？ | 改的是计算图、自动求导、并行切分、状态管理或执行调度 |
| Infra-L4 服务与模型优化 | vLLM、SGLang、TensorRT-LLM、量化、KV Cache、Serving 调度 | 一个模型实例如何高效执行？ | 改的是模型加载、请求处理、缓存、实例吞吐和延迟 |
| Infra-L5 平台与 MLOps | 资源调度、模型仓库、灰度发布、监控、告警、工作流 | 多个模型和用户如何稳定交付？ | 改的是资源编排、版本生命周期、流量治理和服务可用性 |

模型、数据和 workload 不是独立的一层，而是运行在这五层之上的负载面：训练主要落在 Infra-L3，推理主要落在 Infra-L4，最终都受 Infra-L1/Infra-L2 的硬件与系统软件约束。

层间存在灰色地带。例如，FlashAttention 的算法思想属于方法层，kernel 实现属于 Infra-L2，服务集成属于 Infra-L4；FSDP / DeepSpeed 属于 Infra-L3，但底层会调用 Infra-L2 的 NCCL，集群资源又由 Infra-L5 管理；量化理论属于算法方法，低比特 kernel 属于 Infra-L2，推理部署属于 Infra-L4，模型版本和发布流程属于 Infra-L5。KV Cache 的数据结构和调度主要在 Infra-L3/Infra-L4，显存容量和带宽受 Infra-L1 约束，监控和扩缩容则属于 Infra-L5。

Profiling 不属于某一个固定层，而是贯穿 Infra-L1–Infra-L5 的证据工具。它把硬件利用率、kernel 时间、框架调度、服务请求和平台资源放到同一条证据链中；因此一个优化结论不能只说“某层变快了”，还要说明它对 `Compute / Memory / Communication / Quality / End-to-End` 的影响。

## Practice 实践级别

项目中的 `Practice-P0~P3` 描述实验需要达到的真实运行深度，不是 Infra 层级：

| Practice 级别 | 含义 |
|:---|:---|
| Practice-P0 | CPU-first 逻辑验证、公式推导或指标模板 |
| Practice-P1 | 单 GPU、本地模型、单机 profiling 或显存实验 |
| Practice-P2 | vLLM / SGLang 等真实 inference backend |
| Practice-P3 | 多 GPU、分布式通信或分布式 serving |

例如，一个项目可以是 `Practice-P1 + Infra-L4`：在单 GPU 上学习服务实例内部的推理调度；也可以是 `Practice-P2 + Infra-L3–Infra-L4`：接入真实 backend，验证运行时与服务层的性能。

## 横向能力轴

五层结构回答“组件位于哪里”，还需要三条横向能力轴回答“代价如何产生”：

- `Compute`：FLOPs、kernel 时间、利用率和计算重叠。
- `Memory`：容量、带宽、数据驻留、缓存和访存次数。
- `Communication`：GPU 间、CPU-GPU 间、节点间传输、同步和拓扑。

Profiling 与 Evaluation 横跨五层：前者负责采集证据，后者负责验证质量、性能、显存、通信和部署结果。每个优化结论至少要说明 `Compute Δ / Memory Δ / Communication Δ / Quality Δ / End-to-End Δ`。

## 从算法到硬件：如何选择纵向方向

五个方向不是五个互相独立的层级，而是在同一套 Infra 栈上向下穿透的不同入口。可以把它理解为盖房子：训练微调负责“房间如何使用”，推理优化负责“请求如何调度”，显存优化负责“有限空间如何分配”，算子优化负责“单个施工环节如何贴合材料”，编译器图优化负责“如何把设计图自动变成施工方案”。这个比喻用于建立问题意识，不代表方向之间存在简单的难度或价值排序。

| 方向 | 主要落点 | 主要问题 | 典型证据 |
|:---|:---|:---|:---|
| 训练微调 | Infra-L3，受 Infra-L1/Infra-L2 约束 | 模型如何学习、数据如何进入训练、参数如何更新 | loss、质量、显存、step time、训练稳定性 |
| 推理优化 | Infra-L3–Infra-L4 | 请求如何经过 prefill、decode、cache 和调度 | TTFT、TPOT、吞吐、并发、端到端延迟 |
| 显存优化 | 横跨 Infra-L1–Infra-L4 | 状态放在哪里，容量、带宽、重算和搬运如何取舍 | peak memory、带宽、吞吐、质量、OOM 边界 |
| 算子优化 | Infra-L2，受 Infra-L1 约束 | kernel、布局、访存和计算如何贴合硬件 | kernel time、occupancy、带宽利用率、端到端收益 |
| 编译器图优化 | Infra-L2–Infra-L3，连接 Infra-L1/Infra-L4 | 如何进行图变换、融合、lowering 和执行调度 | 编译日志、算子数、kernel 组合、端到端收益 |

选择方向时先问“问题发生在哪一层”，不要先问“哪个方向更热门”：训练问题优先看 Infra-L3，服务延迟优先看 Infra-L3/Infra-L4，OOM 优先看 Memory 轴，单 kernel 热点再下沉到 Infra-L2，图到硬件的映射问题则进入编译与图优化。

## 算子、异构并行与 MLSys 的专题占位

这三类内容不是额外堆出的新主线，而是连接五层结构的横向能力：

| 能力 | 主要连接 | 在专题中的展开位置 | 当前项目入口 |
|:---|:---|:---|:---|
| 算子与编译优化 | Infra-L1–Infra-L3 | 编译与图优化专题；推理专题解释 kernel 对 prefill/decode 的影响 | `66 / 67 / 74` |
| 异构并行与通信 | Infra-L1–Infra-L5 | 通信与并行专题；Profiling 负责定位计算、内存、通信等待 | `79 / 80 / 81` |
| MLSys 方法 | Infra-L2–Infra-L5 | 作为跨专题方法：约束建模、profiling、benchmark、资源调度和回归决策 | `74 / 75 / 79` |

通信库、算子库和分布式并行库也按这个原则放置：NCCL 主要是 Infra-L2 的通信原语，cuBLAS、FlashAttention 等属于 Infra-L2 的算子/内核实现，PyTorch、FSDP、DeepSpeed、Megatron 等属于 Infra-L3 的框架与并行运行时；它们在 Infra-L4 的服务和 Infra-L5 的资源调度中被组合使用。专题不重复介绍同一个库，而是分别解释它在当前问题中的作用和代价。

## 多专题项目如何阅读

一个项目可以被多个专题复用，但只保留一个主叙事入口。主专题负责定义项目问题和最终结论，关联专题只复用其中的指标、机制或实验结果；例如 `66` 的主专题是推理优化，但量化、编译、显存和 Profiling 专题可以分别解释它的低比特、kernel、预算和证据视角。项目资产表中的“主专题 / 关联专题 / Infra 层”用于记录这种关系，避免把同一个项目误读成多个独立项目。

## 横切支撑专题

这些专题不替代主路线，而是把跨路线反复出现的方法轴单独拉出来：

| 专题 | 主入口 | 更适合什么时候进入 |
|:---|:---|:---|
| 量化与压缩 | [quantization](./quantization/intro.md) | 当你同时要看精度、显存、带宽和部署取舍时 |
| 通信与并行 | [communication_parallel](./communication_parallel/intro.md) | 当你开始进入多卡训练、并行切分和通信瓶颈时 |
| Profiling | [profiling](./profiling/intro.md) | 当你需要拿证据，而不是只靠经验猜测时 |
| 后训练与对齐 | [post_training_alignment](./post_training_alignment/intro.md) | 当你完成 SFT 主线后，准备进入对齐与偏好优化时 |

## 基础支撑专题

这些专题更偏机制解释和背景支撑，常作为主路线的前置桥：

| 专题 | 主入口 | 更常服务哪条路线 |
|:---|:---|:---|
| 反向传播与训练机制 | [backpropagation_training_mechanism](./backpropagation_training_mechanism/intro.md) | 训练微调、显存优化 |
| 大模型架构 | [model_architecture](./model_architecture/intro.md) | 训练微调、推理优化 |
| 编译与图优化 | [compiler_graph_optimization](./compiler_graph_optimization/intro.md) | 推理优化、系统优化 |

## 按问题进入专题

如果你已经在做项目，只是遇到瓶颈，再按问题进入横切支撑专题；如果需要补机制背景，再回看基础支撑专题。

常见跳转：

- `结构前置 / SFT / LoRA / 实验收口` -> [监督微调专题](./fine_tuning_training/intro.md)
- `prefill / decode / PagedAttention / benchmark` -> [推理优化专题](./inference_optimization/intro.md)
- `VRAM / checkpoint / offload / memory trade-off` -> [显存优化专题](./memory_performance_tuning/intro.md)
- `量化是否值得做` -> [量化与压缩专题](./quantization/intro.md)
- `为什么慢、为什么爆显存、证据怎么拿` -> [Profiling 专题](./profiling/intro.md)
- `多卡通信和并行切分怎么判断` -> [通信与并行专题](./communication_parallel/intro.md)
