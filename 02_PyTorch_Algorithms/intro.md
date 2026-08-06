# Part 02: PyTorch Algorithm Practice | 第二部分：PyTorch 算法实战

## Part Overview | Part 概览

本部分聚焦 PyTorch 级别的大模型实现，位于 Part 0 / Part 1 之后、Part 3 之前，目标是把基础算子、模型组装、训练与对齐、显存优化、推理优化、并行策略和项目实战串成一条可运行的工程链。正文默认 notebook-first，组页负责组级资产与阅读路径，Part 级导学只管组间关系和阅读顺序，不下沉到具体节号。

`2.7` 已拆成两条子线入口：`2.7A` 负责高级推理策略，`2.7B` 负责模型压缩与量化；总入口页负责说明两条子线如何衔接到 `2.8` 并行通信和 `2.9` 项目验证。

Part 02 现在还配套了两条基础横向机制线和两条方法横向线：
- `反向传播与训练机制`：串起 `00 / 17 / 18 / 12 / 19 / 42 / 74`
- `后训练与对齐`：串起 `14-16 / 50 / 84-85`
- `大模型结构和原理`：承接 `01-08`
- `训练微调闭环`：承接 `09-13 + 60`

Part 2 更像一张多入口学习地图：不同基础和目标的读者可以从不同组切入，最后都汇到项目实战，再按需要回补前面的训练、推理和并行内容。

## 实际演进到的定位（基于 00-86）

```text
Part 02 实际定位：
  大模型算法工程师的 "算法实现底座"
  ├── 组件层（00-08）：PyTorch 实现的核心算子与结构
  ├── 训练层（09-16）：SFT、LoRA、DPO、GRPO 的完整训练链路
  ├── 优化层（17-29）：显存优化、推理加速、并行策略的 PyTorch 实现
  ├── 进阶层（30-59）：前沿算法（长上下文、LoRA 变体、PD 分离）的 PyTorch 实现
  └── 项目层（60-86）：所有算法的完整项目落地
```

关键变化：Part 02 不再是 "PyTorch 入门"，而是 "所有能用 PyTorch / Triton 表达的算法，都在这里实现"。

## Part 02、Part 03、Part 04 的分工

这是理解 86 个文件是否合理的核心框架。

| Part | 名称 | 核心问题 | 抽象层级 | 受众 |
|:---|:---|:---|:---|:---|
| Part 00 | Prerequisites | "Python/PyTorch 怎么用？" | 语言/框架基础 | 所有人 |
| Part 01 | Hardware, Math & Systems | "硬件上发生了什么？" | 硬件/系统原理 | 系统工程师 |
| Part 02 | PyTorch Algorithms | "算法怎么用代码表达？" | 算法实现 | 算法/工程开发 |
| Part 03 | Triton Kernels | "算子怎么写得更快？" | 高性能算子 | 系统/性能工程师 |
| Part 04 | CUDA & System Optimization | "系统怎么极致优化？" | 极致系统优化 | 系统工程师 |

### 关键边界

```text
Part 02（算法实现）：回答 "这个算法在 PyTorch 里怎么写？"
    ↓ 当 PyTorch 不够快时
Part 03（Triton 算子）：回答 "这个算子怎么用 Triton 写得更快？"
    ↓ 当 Triton 不够底层 / 需要极致优化时
Part 04（CUDA / 系统）：回答 "这个系统怎么在 CUDA 层面做极致优化？"
```

## Part 02 编号索引

编号到 86 本身不是问题，但需要让学习者在进入 Part 02 时不被编号淹没。

| 编号区间 | 主题 | 对应路线 |
|:---|:---|:---|
| `00-08` | PyTorch 基础 + 模型组件 | 训练基础 |
| `09-16` | 训练闭环 + 对齐理论 | 训练基础 + 对齐 |
| `17-29` | 显存优化 + 推理加速 + 并行 | 推理 + 显存 + 分布式 |
| `30-59` | 进阶算法（长上下文、LoRA 变体、前沿推理策略） | 各路线进阶 |
| `60-86` | 项目实战 | 所有路线收口 |

### Non-Project Placeholder Map | 非项目占位图

| 编号 | 归属路线 | 预留标题 | 状态 |
|:---|:---|:---|:---|
| `30` | 训练与微调基础 | `Long_Context_Fine_Tuning` | 占位 |
| `31` | 训练与微调基础 | `LoRA_Variants_Theory` | 占位 |
| `32` | 训练与微调基础 | `Data_Engineering_for_SFT` | 占位 |
| `33` | 训练与微调基础 | 预留 | 占位 |
| `34` | 推理优化 | `Prefix_Caching_and_Chunked_Prefill` | 已落盘 |
| `35` | 推理优化 | `Multi_Token_Decoding` | 已落盘 |
| `36` | 推理优化 | `Decode_Scheduling` | 已落盘 |
| `37` | 推理优化 | `KV_Cache_Scheduling` | 已落盘 |
| `38` | 推理优化 | `Prefill_Decode_Disaggregation` | 占位 |
| `39` | 推理优化 | 预留 | 占位 |
| `40` | 显存优化 | `GPTQ_and_AWQ_Weight_Quantization` | 已落盘 |
| `41` | 显存优化 | `FP8_and_KV_Cache_Quantization` | 已落盘 |
| `42` | 显存优化 | `Activation_Offload` | 已落盘 |
| `43` | 显存优化 | `Unified_Memory_Management` | 占位 |
| `44` | 显存优化 | `Auto_Tuning_Framework` | 占位 |
| `45` | 显存优化 | 预留 | 占位 |
| `46` | 通信与并行 | `Communication_Profiling_with_NCCL` | 已落盘 |
| `47` | 通信与并行 | `MoE_Expert_Parallel` | 已落盘 |
| `48` | 通信与并行 | `Communication_Reserved` | 占位 |
| `49` | 通信与并行 | `Parallelism_Reserved` | 占位 |
| `50` | 后训练与对齐 | `Preference_Data_and_Evaluation` | 已落盘 |
| `51` | 后训练与对齐 | `Online_DPO` | 占位 |
| `52` | 后训练与对齐 | `Alignment_Reserved` | 占位 |
| `53` | 通用预留 | `Reserved_53` | 占位 |
| `54` | 通用预留 | `Reserved_54` | 占位 |
| `55` | 通用预留 | `Reserved_55` | 占位 |
| `56` | 通用预留 | `Reserved_56` | 占位 |
| `57` | 通用预留 | `Reserved_57` | 占位 |
| `58` | 通用预留 | `Reserved_58` | 占位 |
| `59` | 通用预留 | `Reserved_59` | 占位 |

### Project Placeholder Map | 项目占位图

| 编号 | 归属路线 | 预留标题 | 状态 |
|:---|:---|:---|:---|
| `60` | 训练与微调基础 | `LoRA_Full_Project` | 已落盘 |
| `61` | 训练与微调基础 | `Model_Architecture_Exploration` | 占位 |
| `62` | 训练与微调基础 | `Instruction_Fine_Tuning_Project` | 占位 |
| `63` | 训练与微调基础 | `LoRA_Variants_Benchmark` | 占位 |
| `64` | 训练与微调基础 | 预留 | 占位 |
| `65` | 训练与微调基础 | 预留 | 占位 |
| `66` | 推理优化 | `Inference_Performance_Comparison` | 已落盘 |
| `67` | 推理优化 | `Quantized_Inference_and_Deployment` | 已落盘 |
| `68` | 推理优化 | `Speculative_Decoding_Benchmark` | 占位 |
| `69` | 推理优化 | `Prefix_Caching_Benchmark` | 占位 |
| `70` | 推理优化 | `Reserved_70` | 占位 |
| `71` | 推理优化 | `Reserved_71` | 占位 |
| `72` | 推理优化 | `Reserved_72` | 占位 |
| `73` | 显存优化 | `Training_Performance_Analysis` | 已落盘 |
| `74` | 显存优化 | `Profiling_Driven_Optimization` | 已落盘 |
| `75` | 显存优化 | `Reserved_75` | 占位 |
| `76` | 显存优化 | `Reserved_76` | 占位 |
| `77` | 显存优化 | `Reserved_77` | 占位 |
| `78` | 显存优化 | `Reserved_78` | 占位 |
| `79` | 通信与并行 | `Distributed_Parallel_Benchmark` | 已落盘 |
| `80` | 通信与并行 | `MoE_Expert_Parallel_Benchmark` | 占位 |
| `81` | 通信与并行 | `Reserved_81` | 占位 |
| `82` | 通信与并行 | `Reserved_82` | 占位 |
| `83` | 通信与并行 | `Reserved_83` | 占位 |
| `84` | 后训练与对齐 | `DPO_Preference_Project` | 已落盘 |
| `85` | 后训练与对齐 | `GRPO_Groupwise_Alignment_Project` | 已落盘 |
| `86` | 后训练与对齐 | `DPO_Online_Benchmark` | 占位 |
| `87` | 通用预留 | 预留 | 占位 |
| `88` | 通用预留 | 预留 | 占位 |
| `89` | 通用预留 | 预留 | 占位 |

## Part Asset Overview | Part 资产总览

本章内容按 9 个主题组组织，后续页面也沿该结构继续扩展。

> 导航说明：先看总览，再进入具体组页。
> 组页负责组内阅读顺序与资产收口，不需要一次性读完全部页面。
> Part 2 既是工程实战目录，也是 Part 0 / Part 1 之后、Part 3 之前的共同衔接层。

| 学习组 | 职责作用 | 当前内容映射 | 每组多少节 |
|:---|:---|:---|:---|
| [2.1](./2_1.md) | 建立基础算子和组件直觉 | [00](./00_PyTorch_Warmup.ipynb)、[01](./01_RMSNorm_Tutorial.ipynb)、[02](./02_SwiGLU_Activation.ipynb)、[03](./03_RoPE_Tutorial.ipynb)、[04](./04_Attention_MHA_GQA.ipynb) | 5 |
| [2.2](./2_2.md) | 组装模型结构并理解 MoE 组件 | [05](./05_LLaMA3_Block_Tutorial.ipynb)、[06](./06_MoE_Router.ipynb)、[07](./07_MoE_Load_Balancing_Loss.ipynb)、[08](./08_Architecture_Tricks.ipynb) | 4 |
| [2.3](./2_3.md) | 搭起微调、调度器和训练闭环 | [09](./09_SFT_Training_Loop.ipynb)、[10](./10_LoRA_Tutorial.ipynb)、[11](./11_LR_Schedulers_WSD_Cosine.ipynb)、[12](./12_Gradient_Accumulation.ipynb)、[13](./13_End_to_End_Fine_Tuning_Experiment.ipynb) | 5 |
| [2.4](./2_4.md) | 理解偏好优化与对齐链路 | [14](./14_RLHF_PPO_Memory.ipynb)、[15](./15_DPO_Loss_Tutorial.ipynb)、[16](./16_GRPO_Loss_Tutorial.ipynb) | 3 |
| [2.5](./2_5.md) | 追踪反向传播和显存优化 | [17](./17_Autograd_Basics.ipynb)、[18](./18_Activation_and_Loss_Backward.ipynb)、[19](./19_Activation_Checkpointing_and_Activation_Offload.ipynb) | 3 |
| [2.6](./2_6.md) | 建立推理加速和缓存直觉 | [20](./20_FlashAttention_Sim.ipynb)、[21](./21_Decoding_Strategies.ipynb)、[22](./22_vLLM_PagedAttention.ipynb) | 3 |
| [2.7](./2_7.md) | 2.7A 高级推理 / 2.7B 压缩量化双轨入口，继续向 serving、cache 和量化家族扩展 | 核心：[23](./23_Speculative_Decoding.ipynb)、[24](./24_SGLang_RadixAttention.ipynb)、[25](./25_Quantization_W8A16.ipynb)、[26](./26_QLoRA_and_4bit_Quantization.ipynb)；扩展：[34](./34_Prefix_Caching_and_Chunked_Prefill.ipynb)、[35](./35_Multi_Token_Decoding.ipynb)、[36](./36_Decode_Scheduling.ipynb)、[40](./40_GPTQ_and_AWQ_Weight_Quantization.ipynb)、[41](./41_FP8_and_KV_Cache_Quantization.ipynb)、[37](./37_KV_Cache_Scheduling.ipynb) | 核心 4 + 扩展 6 |
| [2.8](./2_8.md) | 形成并行策略和通信边界判断，并延伸到通信 profiling | 核心：[27](./27_ZeRO_Optimizer_Sim.ipynb)、[28](./28_Pipeline_Parallelism_MicroBatch.ipynb)、[29](./29_Tensor_Parallelism_Sim.ipynb)；扩展：[46](./46_Communication_Profiling_with_NCCL.ipynb) | 核心 3 + 扩展 1 |
| [2.9](./2_9.md) | 汇总项目验证和工程闭环，承接训练 / 推理 / 系统 / 部署项目 | 核心：[60](./60_LoRA_Fine_Tuning_Project.ipynb)、[66](./66_Inference_Performance_Comparison.ipynb)、[73](./73_Training_Performance_Analysis.ipynb)；扩展：[74](./74_Profiling_Driven_End_to_End_Optimization.ipynb)、[79](./79_Distributed_Parallel_Benchmark.ipynb)、[67](./67_Quantized_Inference_and_Deployment.ipynb) | 核心 3 + 扩展 3 |

## Learning Path | 学习路径

Part 2 可以按多条入口理解：零基础入口先把算子、组装、训练与项目闭环串起来；训练优先、推理优先和并行优先入口则可以从不同工程目标切入，最后都回到项目实战。

### Recommended Order | 推荐顺序

- 零基础入口：先看 [2.1](./2_1.md) -> [2.2](./2_2.md) -> [2.3](./2_3.md) -> [2.5](./2_5.md) -> [2.9](./2_9.md)
- 训练优先入口：先看 [2.3](./2_3.md) -> [2.4](./2_4.md) -> [2.5](./2_5.md) -> [2.9](./2_9.md)
- 对齐优先入口：先看 [2.3](./2_3.md) -> [2.4](./2_4.md) -> [2.9](./2_9.md)
- 推理优先入口：先看 [2.6](./2_6.md) -> [2.7](./2_7.md) -> [2.9](./2_9.md)
- 并行优先入口：先看 [2.8](./2_8.md) -> [2.9](./2_9.md)
- 系统学习：按 [2.1](./2_1.md) -> [2.2](./2_2.md) -> [2.3](./2_3.md) -> [2.4](./2_4.md) -> [2.5](./2_5.md) -> [2.6](./2_6.md) -> [2.7](./2_7.md) -> [2.8](./2_8.md) -> [2.9](./2_9.md) 顺序推进

### Next Steps | 后续衔接

- 基础认知层：先看 [2.1](./2_1.md)、[2.2](./2_2.md)，把基础算子和模型组装先立住，再按需要进入 [2.5](./2_5.md)。
- 训练与对齐层：先看 [2.3](./2_3.md)、[2.4](./2_4.md)、[2.5](./2_5.md)，把训练、对齐和显存优化的链路理顺，主要衔接后续实现页和项目页。
- 推理与并行层：先看 [2.6](./2_6.md)、[2.7](./2_7.md)、[2.8](./2_8.md)，把推理、压缩和并行策略串起来，主要衔接项目实战与后续实现页。
- 项目收口：最后看 [2.9](./2_9.md)，把前面的知识点放回真实项目里验证和收束。

## Environment Notes | 环境说明

- 默认按 `CPU-first` 设计
- 这里只写 Part 级统一前提，不点到具体节号
- 少数 notebook 如需 `GPU optional`、`GPU required` 或多卡/完整工具链，以单页说明为准，不在导学页重复展开
