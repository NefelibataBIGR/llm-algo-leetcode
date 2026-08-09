# 04 Attention Evolution

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

它的意义在于：

- 让不同 head 可以学习不同的关系模式
- 在表达能力上提供足够大的自由度
- 作为后续所有优化版本的参照系

但 MHA 的代价也很明显：head 数越多，训练和推理时的算力与访存压力越大。

### MQA / GQA

为了降低推理时 KV cache 和带宽成本，多个 query head 可以共享更少的 key/value head。

这类设计的核心目标是：

- 降低推理显存
- 提升吞吐
- 尽量保留多头建模能力

MQA / GQA 的演化反映的是非常现实的工程权衡：

- query head 仍然需要多样性
- 但 key/value 并不一定要为每个 head 完全复制
- 在不显著伤害效果的前提下，cache 和带宽成本可以明显下降

这也是为什么很多现代模型会在“结构表达能力”和“推理成本”之间选择折中方案，而不是死守原始 MHA。

### MLA：低秩潜在注意力

在 MQA / GQA 之后，注意力演化开始进一步压缩 KV 表示本身。

MLA 可以理解为一种更激进的 KV 压缩路线：

- 它不直接缓存所有 head 的完整 KV
- 而是先把 KV 投到更低维的 latent 空间
- 推理时只维护压缩后的 latent cache，再在需要时恢复或重构注意力所需的信息

这条路线的核心收益有两个：

- 显著降低 KV cache 占用和带宽压力
- 在很多场景下保持接近甚至优于传统 MHA 的效果

所以，MLA 不是简单的“再少几个 head”，而是把 attention 的缓存表示重新设计了一遍。

### 稀疏 / 长上下文 attention

- 一部分方法通过局部窗口、分块或路由减少计算
- 一部分方法通过系统优化改善吞吐和缓存
- 一部分方法通过结构改造把 attention 的成本压低

在 DeepSeek 的演进里，这条线又继续分成了两类思路：

- `DSA` 这类 token-level sparse attention：先粗筛再精读
- `NSA` 这类硬件对齐的 sparse attention：把稀疏模式和 GPU 友好执行绑在一起

对长文本而言，这意味着模型不再“通读全文”，而是先判断哪些 token 值得看，再把算力集中到真正重要的部分。

当上下文拉长后，attention 的问题不再只是“算得慢”，而是“算得起吗”：

- 全局 attention 的二次复杂度会迅速放大
- KV cache 的常驻成本会挤压 batch 和并发
- 不同任务对局部依赖和全局依赖的需求并不相同

因此，稀疏、局部和分块方案本质上是在重新定义“哪些 token 真的需要互相看见”。

### DeepSeek 风格的索引-选择式注意力

DeepSeek 的稀疏注意力演化，尤其适合用“先翻目录、再精读”来理解。

可以把它拆成两个阶段：

- `Lightning Indexer`：快速给历史 token 打相关性分数
- `Selector`：只保留 Top-k token 进入后续的精细注意力

这样做的目标是把复杂度从 `O(L²)` 压到更接近 `O(L·k)`，尤其适合长上下文解码。

如果再细分实现，DeepSeek 系列的稀疏注意力可以理解成三条并行路径：

- 压缩分支：抓大意，处理长程概览
- 选择分支：对高相关 token 做精读
- 滑动窗口分支：保留局部细节和邻域信息

这类设计的关键不是“把 attention 变稀疏”这么简单，而是把全局、局部和选择性关注拆成不同通路，再让硬件执行尽可能顺滑。

### 系统级加速

FlashAttention 这类工作不是重新定义 attention 语义，而是重新定义执行方式。

这类工作的关键是把理论计算图翻译成更接近硬件的执行路径：

- 减少不必要的 HBM 读写
- 尽量在更合适的粒度上做分块和融合
- 让 attention 的瓶颈尽量从访存转向算力本身

所以 attention 的演化实际上分成两条线：

- 一条是结构语义上的演化，例如 MHA 到 GQA
- 一条是系统实现上的演化，例如 FlashAttention 这类 IO-aware 优化

现代 LLM 往往同时吃这两条收益。

## 代表模型

- `LLaMA`：以标准 attention 为基础，再通过 GQA 等设计优化成本
- `Mistral`：会把局部窗口和长上下文设计结合起来看
- `DeepSeek`：更激进地重构 attention 结构，适合作为现代注意力演化的代表案例，尤其适合看 MLA 和 sparse attention

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
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 看 MLA 如何把 KV cache 压缩到更低维的 latent 表示。 |
| [Mistral 7B](https://arxiv.org/abs/2310.06825) | 代表将滑动窗口等机制融入主流 attention 的实践路线。 |
| [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) | 看 DeepSeek 如何把稀疏注意力推进到 DSA 路线。 |
| [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) | 看硬件对齐的稀疏注意力如何把局部、压缩和选择性关注组合起来。 |

## 与 Part 02 的对应关系

- `04` 直接讲 MHA / GQA / MQA 的 head 关系
- `05` 里可以看到 attention 如何放进 block
- `08` 里可以看到真实模型中的 attention 变体
- `22`、`24`、`67` 等页面和这里形成系统侧衔接
- `09` 里会看到稀疏化和路由在更大结构中的位置

## 可视化提示

建议画两张图：

- 一张 `MHA -> MQA -> GQA` 的 head 关系图
- 一张 `MHA -> GQA -> MLA -> sparse attention` 的演化图
- 一张 attention 成本图，标出训练计算、推理 KV cache 和系统吞吐之间的关系

## 阅读建议

如果你要继续扩展，建议接着看：

- `05_rope_position_encoding.md`
- `06_block_residual_path.md`
- `08_representative_models.md`
