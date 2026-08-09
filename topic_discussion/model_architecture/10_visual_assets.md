# 10 Visual Assets

## 页面目标

这一页不讲新知识，只沉淀图。

它的作用是把前面所有专题页的内容压缩成可视化入口，帮助读者快速定位：

- 现在讨论的是哪一代结构
- 这个模块在 block 里放在哪
- 哪个模型选了哪种结构
- 哪些设计是为了训练，哪些是为了推理，哪些是为了显存

## 资产清单

### 1. 模块演进时间线

建议做一张总时间线，把这些模块按演进顺序连起来：

- Tokenization / BPE / Embedding
- LayerNorm -> Pre-Norm -> RMSNorm
- MHA -> MQA -> GQA -> 稀疏 / 长上下文 attention
- 绝对位置编码 -> 相对位置编码 -> RoPE -> RoPE 扩展
- FFN -> GELU -> SwiGLU -> MoE-FFN

这张图的作用是让读者先知道“演化顺序”，再去看每个单页。

![模块演进时间线](/topic_discussion/model_architecture/timeline.png)

### 2. Block 总图

建议画一张现代 decoder block 图，至少包含：

- input hidden state
- first norm
- self-attention
- residual add
- second norm
- MLP / SwiGLU
- residual add
- optional MoE replacement

这张图是 `06_block_residual_path.md` 的视觉版本，也是 `01-08` 的总对照图。

![Decoder Block 总图](/topic_discussion/model_architecture/block_overview.png)

### 3. 代表模型结构矩阵

建议画一张表或矩阵图，按行列展示：

- 行：LLaMA / Mistral / Qwen / Gemma / DeepSeek
- 列：tokenization、norm、attention、RoPE、MLP、decoder-only、长上下文、系统优化

它的作用不是记参数，而是快速看出各模型的结构选择差异。

这张矩阵建议后续用表格 + 图结合呈现；当前已经补了一版视觉锚点：

![代表模型结构矩阵](/topic_discussion/model_architecture/representative_models_matrix.svg)

### 4. 跨模块关系图

建议画一张关系图，把这些页面之间的依赖连起来：

- `02_tokenization_embedding` -> `06_block_residual_path`
- `03_norm_evolution` -> `06_block_residual_path`
- `04_attention_evolution` -> `05_rope_position_encoding`
- `07_mlp_ffn_evolution` -> `06_block_residual_path`
- `01_transformer_decoder` -> `08_representative_models`
- `08_representative_models` -> `09_moe_sparsity_evolution`

这张图是整个专题的“知识导航图”。

专题页之间的关系图，当前已经补了一版：

![跨模块知识地图](/topic_discussion/model_architecture/cross_module_map.svg)

## 当前图册

建议的观看顺序：

1. `专题总导航图`
2. `模块演进时间线`、`Decoder Block 总图`
3. `DeepSeek Attention 演进图`、`MoE / Sparsity 路由图`
4. `MoE / Sparsity 演进图`
5. `代表模型结构矩阵`、`跨模块知识地图`

- [专题总导航图](/topic_discussion/model_architecture/overview_map.svg)
- [模块演进时间线](/topic_discussion/model_architecture/timeline.png)
- [Decoder Block 总图](/topic_discussion/model_architecture/block_overview.png)
- [DeepSeek Attention 演进图](/topic_discussion/model_architecture/deepseek_attention.png)
- [MoE / Sparsity 路由图](/topic_discussion/model_architecture/moe_sparsity.png)
- [MoE / Sparsity 演进图](/02_PyTorch_Algorithms/09_moe_sparsity_evolution.svg)
- [代表模型结构矩阵](/topic_discussion/model_architecture/representative_models_matrix.svg)
- [跨模块知识地图](/topic_discussion/model_architecture/cross_module_map.svg)

## 设计原则

- 图不追求花哨，追求信息密度
- 每张图只回答一个核心问题
- 读者看图后应该能知道下一页该去哪
- 图要和正文页一一对应，不要单独悬空

## 与 Part 02 的对应关系

这页不是独立知识点，而是给 `Part 02 01-08` 和后续扩展页配图：

- `01`、`02`、`03`、`04`、`05` 可以统一落到 block 总图
- `06`、`07` 可以统一落到 MoE / FFN 结构图
- `08` 可以统一落到模型结构矩阵和真实实现对照图

## 阅读建议

如果你已经看过：

- `02_tokenization_embedding.md`
- `03_norm_evolution.md`
- `04_attention_evolution.md`
- `05_rope_position_encoding.md`
- `01_transformer_decoder.md`
- `07_mlp_ffn_evolution.md`
- `06_block_residual_path.md`
- `08_representative_models.md`
- `09_moe_sparsity_evolution.md`

那么这一页就是把它们全部收束成图的地方。
