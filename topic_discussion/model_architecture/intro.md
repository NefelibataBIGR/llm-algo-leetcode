# 大模型架构专题

## 专题定位

本专题用于串起大模型结构主线：先看 token 怎样进入 block，再看 norm、attention、RoPE、MLP、MoE 和结构技巧分别放在什么位置，最后把这些结构差异接回训练、推理和显存路线。这里聚焦“组件在 block 里怎么组织、为什么这样组织”；如果问题已经转到训练微调、推理服务或显存 trade-off，应转到对应主路线专题。

## 主学习线

`Task1-6` 是学习路线，指向 `Part02` 和相关扩展页的具体小节；最后一列主要对应 `01-09` 的专题正文页，`10` 是图册补充。

| Task | 学习内容 | 主学习线 | 专题正文 |
|:---|:---|:---|:---|
| Task1 | token、embedding 与 norm 前置 | `01 -> 02` | [02 Tokenization and Embedding](./02_tokenization_embedding.md), [03 Norm Evolution](./03_norm_evolution.md) |
| Task2 | RoPE 与 attention 结构 | `03 -> 04` | [04 Attention Evolution](./04_attention_evolution.md), [05 RoPE Position Encoding](./05_rope_position_encoding.md) |
| Task3 | block 组装与 residual 主干 | `05` | [01 Transformer Decoder](./01_transformer_decoder.md), [06 Block Residual Path](./06_block_residual_path.md) |
| Task4 | MLP / SwiGLU 演化 | `02 -> 08` | [07 MLP FFN Evolution](./07_mlp_ffn_evolution.md) |
| Task5 | MoE、router 与稀疏化 | `06 -> 07` | [09 MoE Sparsity Evolution](./09_moe_sparsity_evolution.md) |
| Task6 | 代表模型与结构对照 | `08` | [08 Representative Models](./08_representative_models.md) |

## 正文与跳转

先按上面的 `Task1-6` 走结构主线；遇到“这些组件到底在 block 的哪一段”“真实模型和教科书结构差在哪”时，再回来看对应的专题正文。想看汇总版就进 [大模型结构和原理正文](./casebook.md)，想按连续故事线走一遍就进 [大模型结构和原理深入阅读](./walkthrough.md)。图册补充放在 [10 Visual Assets](./10_visual_assets.md)。

如果问题已经跨到别的专题：
[监督微调专题](../fine_tuning_training/intro.md) 负责 LoRA 和训练闭环，[推理优化专题](../inference_optimization/intro.md) 负责 attention、KV cache 和服务链路，[显存优化专题](../memory_performance_tuning/intro.md) 负责结构带来的资源代价。
