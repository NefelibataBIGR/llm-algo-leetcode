# 大模型架构专题

## 专题定位与 Infra 层关系

本专题串起大模型结构主线：先看 token 怎样进入 block，再看 norm、attention、RoPE、MLP、MoE 和结构技巧分别放在什么位置，最后把这些结构差异接回训练、推理和显存路线。模型架构是运行在五层 Infra 之上的负载面，不等同于某一个软件层：结构决定计算图、参数规模、激活和 KV Cache 形态，Infra-L1–Infra-L3 决定它如何被执行，Infra-L4 决定它如何被服务，Infra-L5 负责评测、发布和部署治理。

本专题属于基础支撑专题，不采用主学习路线的六节正文结构；正文按 `01-09` 组织，`10_visual_assets.md` 作为图册补充。阅读时应把结构判断落回计算、内存、通信和质量指标；如果问题已经转到训练微调、推理服务或显存 trade-off，应转到对应主路线专题。

## 推荐入口

推荐从 [Part 02 导学](../../02_PyTorch_Algorithms/intro.md) 的结构与训练前置进入；如果目标是推理，可从 attention、RoPE、KV cache 相关小节切入。需要完整结构验证时，再连接 `61` 模型架构探索项目。

## 前置阅读

建议先具备 `Part 00` 的张量和模块基础，再回看 Part 02 的 `01-04` 组件 notebook。`05` 和 `08` 适合作为 block 组装与架构变体的收束，MoE 相关内容按需扩展即可。

## 主学习线

`Task1-6` 是学习路线，指向 `Part 02` 和相关扩展页的具体小节；最后一列主要对应 `01-09` 的专题正文页，`10` 是图册补充。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | decoder-only 总览与 token、embedding 入口 | `01 -> 02` | [01 Transformer Decoder](./01_transformer_decoder.md), [02 Tokenization and Embedding](./02_tokenization_embedding.md) |
| Task2 | norm、attention 与位置编码 | `03 -> 04 -> 05` | [03 Norm Evolution](./03_norm_evolution.md), [04 Attention Evolution](./04_attention_evolution.md), [05 RoPE Position Encoding](./05_rope_position_encoding.md) |
| Task3 | block 组装与 residual 主干 | `06` | [06 Block Residual Path](./06_block_residual_path.md) |
| Task4 | MLP / SwiGLU 演化 | `07` | [07 MLP FFN Evolution](./07_mlp_ffn_evolution.md) |
| Task5 | MoE、router 与稀疏化 | `09` | [09 MoE Sparsity Evolution](./09_moe_sparsity_evolution.md) |
| Task6 | 代表模型、结构对照与架构验证项目 | `08 -> 61` | [08 Representative Models](./08_representative_models.md) |

## 正文与跳转

先按上面的 `Task1-6` 走结构主线；遇到“这些组件到底在 block 的哪一段”“真实模型和教科书结构差在哪”时，再回来看对应的专题正文。想看汇总版就进 [大模型结构和原理正文](./casebook.md)，想按连续故事线走一遍就进 [大模型结构和原理深入阅读](./walkthrough.md)。图册补充放在 [10 Visual Assets](./10_visual_assets.md)。

如果问题已经跨到别的专题：
[监督微调专题](../fine_tuning_training/intro.md) 负责 LoRA 和训练闭环，[推理优化专题](../inference_optimization/intro.md) 负责 attention、KV cache 和服务链路，[显存优化专题](../memory_performance_tuning/intro.md) 负责结构带来的资源代价。

## 环境与验证

结构阅读、参数统计和大多数组件实验可先用 CPU；若要比较真实模型的显存、吞吐或服务行为，需要 GPU 和对应推理后端。架构结论应回到参数量、计算量、显存、质量或延迟等可观测指标，不能只凭结构图判断优劣。
