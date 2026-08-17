# 06 Transformer Decoder

## 页面目标

这一页解释为什么现代大语言模型大多采用 decoder-only 架构，以及这种结构和训练、推理、上下文建模之间的关系。

## 问题起点

Transformer 不是只有 encoder 和 decoder 两条分支，但在大模型时代，decoder-only 成了最常见的主干选择。

这一选择背后的原因通常是：

- 生成任务天然需要自回归解码
- decoder-only 更直接地对齐 next-token prediction
- 结构上更容易和推理系统、KV cache 和采样循环连接

## 演化过程

### Encoder-Decoder 时代

早期 seq2seq 任务通常依赖 encoder-decoder 结构，适合翻译和条件生成。

### Decoder-only 时代

大模型预训练逐渐收敛到自回归目标，decoder-only 结构更自然地对齐语言建模。

### 工程化的 decoder-only

现代 decoder-only 模型通常结合：

- causal mask
- pre-norm block
- KV cache
- 位置编码

这些因素一起决定了它在训练和推理中的表现。

## 代表模型

- `GPT` 系列：decoder-only 路线的代表
- `LLaMA`：现代开源 decoder-only block 的典型样本
- `DeepSeek`：在 decoder-only 主干上继续做结构和效率优化

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer 的起点，理解 encoder-decoder 和 decoder 结构的基础。 |
| [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) | GPT 路线的重要入口，理解 decoder-only 预训练范式。 |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | 说明 decoder-only 模型如何自然延伸到指令微调和对齐。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 现代 decoder-only block 的高质量参考实现。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 代表 decoder-only 结构与局部窗口、长上下文结合的实践。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 代表 decoder-only 主干上更激进的结构优化。 |

## 与 Part 02 的对应关系

- `05` 的 LLaMA Block 直接落在 decoder-only 主干上
- `04`、`03`、`01` 的组件都服务于 decoder-only block
- `22`、`24`、`67` 等推理页讨论的 KV cache 和 decode loop 也依赖这个结构

## 可视化提示

建议画一张 `encoder-decoder` 到 `decoder-only` 的对比图，标出：

- 输入路径
- causal mask
- self-attention
- next-token generation loop

## 阅读建议

如果你要继续扩展，建议接着看：

- `mlp_ffn_evolution.md`
- `block_residual_path.md`
- `representative_models.md`
