# 08 Representative Models

## 页面目标

这一页不重复讲模块本身，而是回答一个更实际的问题：

- 当这些模块组合起来时，不同流行大模型到底选了什么结构
- 它们在 norm、attention、RoPE、MLP、decoder 结构上有什么差异
- 为什么这些差异值得单独学

## 问题起点

模块演进最终会落到具体模型上。

如果只学模块，不看模型，容易把知识停留在“概念正确”；
如果只看模型，不拆模块，又容易把结构选择混成一个黑箱。

这一页的作用，就是把两者接起来。

## 演化过程

### LLaMA：现代开源 block 的参照系

LLaMA 通常可以看成现代 dense decoder-only block 的参照样本：

- RMSNorm
- RoPE
- SwiGLU
- decoder-only 主干

它的价值不只是“很流行”，而是它把现代 LLM block 的常见选择收敛成了一条清晰路径。

### Mistral：面向推理和长上下文的效率优化

Mistral 更适合用来观察：

- 局部窗口 attention
- 长上下文效率
- decoder-only 结构与系统成本的关系

### Qwen：工程可用性和多语言能力

Qwen 更适合用来观察：

- tokenizer 和 embedding 的覆盖面
- 长上下文和工程部署策略
- 结构设计如何服务多语言任务

### Gemma：紧凑而稳定的现代结构样本

Gemma 常被用来观察：

- norm 和 block 的稳定性
- 结构简洁性
- 现代模型如何在效果和成本之间折中

### DeepSeek：结构重构更激进的代表

DeepSeek 适合作为更前沿的对照样本，尤其是：

- attention 结构的重构
- 长上下文与效率优化
- decoder-only 主干上的进一步演化

## 代表模型对照

| 模型 | 你应该关注什么 |
|:---|:---|
| LLaMA | 现代标准 block 是怎么收敛出来的 |
| Mistral | attention 和长上下文怎么结合 |
| Qwen | tokenizer、embedding 和工程可用性 |
| Gemma | norm、MLP、block 的稳定性 |
| DeepSeek | attention / block 结构还能怎么重构 |

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 block 的基准模型。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 长上下文和效率优化的重要样本。 |
| [Qwen Technical Report](https://arxiv.org/abs/2309.16609) | 多语言和工程实践的重要样本。 |

## 前沿论文

## 可视化提示

建议在这一页放两张图，把“模型选择”和“横向对照”分开：

![代表模型结构矩阵](/topic_discussion/model_architecture/representative_models_matrix.svg)

![跨模块知识地图](/topic_discussion/model_architecture/cross_module_map.svg)

第一张图负责快速看清 LLaMA、Mistral、Qwen、Gemma、DeepSeek 各自的结构选择；第二张图负责把 01-09 页串成一张知识导航图。

| 文献 | 读它的理由 |
|:---|:---|
| [Gemma: Open Models Based on Gemini Research and Technology](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) | 现代紧凑模型的参考样本。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 更激进的结构重构样本。 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 更前沿的系统化结构演化样本。 |

## 与 Part 02 的对应关系

- `01-08` 是这些模型结构设计的底层来源
- `05` 直接对应 decoder block 的核心样式
- `04`、`03`、`02`、`01` 决定这些模型 block 具体怎么搭
- `06 / 07` 对应 MoE 或稀疏结构扩展

## 可视化提示

建议做一张“代表模型结构矩阵”：

- 行：LLaMA / Mistral / Qwen / Gemma / DeepSeek
- 列：norm、attention、RoPE、MLP、decoder-only、长上下文、系统优化

这样读者能快速看出不同模型的结构选择差异。

## 阅读建议

如果你已经看过：

- `tokenization_embedding.md`
- `norm_evolution.md`
- `attention_evolution.md`
- `rope_position_encoding.md`
- `transformer_decoder.md`
- `mlp_ffn_evolution.md`
- `block_residual_path.md`

这一页就是把它们投射到真实模型上的地方。
