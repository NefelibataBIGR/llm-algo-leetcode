# 03 Attention Evolution

## 页面目标

这一页解释 attention 为什么从标准 MHA 演化到 MQA、GQA，以及为什么后续又出现稀疏 attention、长上下文 attention 和系统级加速设计。

## 问题起点

Attention 解决的是 token 之间如何建立依赖关系的问题，但它同时也是：

- 训练里的主要算力和访存来源之一
- 推理时 KV cache 的核心来源之一
- 长上下文和并发场景里的主要瓶颈之一

所以 attention 的演化，本质上是在优化“表达能力 vs 成本”。

## 演化过程

### 标准 MHA

多头注意力把不同子空间的关系拆开建模，是最基础的现代 attention 形式。

### MQA / GQA

为了降低推理时 KV cache 和带宽成本，多个 query head 可以共享更少的 key/value head。

这类设计的核心目标是：

- 降低推理显存
- 提高吞吐
- 尽量保留多头建模能力

### 稀疏 / 长上下文 attention

- 一部分方法通过局部窗口、分块或路由减少计算
- 一部分方法通过系统优化改善吞吐和缓存
- 一部分方法通过结构改造把 attention 的成本压低

### 系统级加速

FlashAttention 这类工作不是重新定义 attention 语义，而是重新定义执行方式。

## 代表模型

- `LLaMA`：以标准 attention 为基础，再通过 GQA 等设计优化成本
- `Mistral`：会把局部窗口和长上下文设计结合起来看
- `DeepSeek`：更激进地重构 attention 结构，适合作为现代注意力演化的代表案例

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | attention 的总起点，理解所有后续变体必须先看它。 |
| [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | MQA 的经典入口，解释为什么推理时可以减少 KV 头。 |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 理解 GQA 如何在表达能力和推理成本之间折中。 |

## 前沿论文

| 文献 | 读它的理由 |
|:---|:---|
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 代表系统层面的 attention 优化，直接影响训练和推理速度。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 代表将滑动窗口等机制融入主流 attention 的实践路线。 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 代表对 attention / KV 表示做进一步重构的前沿模型案例。 |

## 与 Part 02 的对应关系

- `04` 直接讲 MHA / GQA / MQA 的 head 关系
- `05` 里可以看到 attention 如何放进 block
- `08` 里可以看到真实模型中的 attention 变体
- `22`、`24`、`67` 等页面和这里形成系统侧衔接

## 可视化提示

建议画两张图：

- 一张 `MHA -> MQA -> GQA` 的 head 关系图
- 一张 attention 成本图，标出训练计算、推理 KV cache 和系统吞吐之间的关系

## 阅读建议

如果你要继续扩展，建议接着看：

- `rope_position_encoding.md`
- `block_residual_path.md`
- `representative_models.md`
