# 专题讨论轴

`topic_discussion` 是跨 `Part00-04` 的知识组织层。它不替代纵向 notebook 主线，只负责三件事：

- 把 `Part02` 的主学习路线收成更清楚的入口
- 给跨路线反复出现的方法轴补判断框架
- 给基础机制补解释层，避免读者只记结论、不懂来源

## 主学习路线

这三条专题直接承接 `Part02` 的主任务带和项目收口带：

| 路线 | 主入口 | 适合什么时候进入 |
|:---|:---|:---|
| 训练微调方向 | [监督微调专题](./fine_tuning_training/intro.md) | 当你要从结构前置、SFT、LoRA 一路走到项目交付时 |
| 推理优化方向 | [推理优化专题](./inference_optimization/intro.md) | 当你要系统理解 prefill、decode、KV cache、服务和 benchmark 时 |
| 显存优化方向 | [显存优化专题](./memory_performance_tuning/intro.md) | 当你要把训练显存、推理 cache、量化预算和 trade-off 串起来时 |

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

## 怎么选入口

- 如果你要完整走 `Part02` 主线，先从三条主学习路线里选一条。
- 如果你已经在做项目，只是遇到瓶颈，再按问题进入横切支撑专题。
- 如果你需要补机制背景，再回看基础支撑专题。

常见跳转：

- `结构前置 / SFT / LoRA / 实验收口` -> [监督微调专题](./fine_tuning_training/intro.md)
- `prefill / decode / PagedAttention / benchmark` -> [推理优化专题](./inference_optimization/intro.md)
- `VRAM / checkpoint / offload / memory trade-off` -> [显存优化专题](./memory_performance_tuning/intro.md)
- `量化是否值得做` -> [量化与压缩专题](./quantization/intro.md)
- `为什么慢、为什么爆显存、证据怎么拿` -> [Profiling 专题](./profiling/intro.md)
- `多卡通信和并行切分怎么判断` -> [通信与并行专题](./communication_parallel/intro.md)
