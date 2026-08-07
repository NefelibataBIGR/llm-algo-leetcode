# 大模型结构和原理深入阅读

## 主故事线

一条典型的结构故事，是让一个 token 的 hidden state 按照真实 block 跑一遍：

`token embedding -> RMSNorm -> Attention -> Residual -> RMSNorm -> SwiGLU / MLP -> Residual -> optional MoE / tricks`

这条线的重点不是“每一步算了什么公式”，而是“张量在 block 里怎么走，为什么这样排”。

## 样板叙事

如果把这个专题写成一个完整的讲述，它应该回答三个问题：

1. 为什么结构理解值得单独成专题
- 因为训练、推理、显存和微调的很多判断，最后都要回到 block 结构本身。

2. 结构演化是怎么发生的
- 从基础的 norm / attention / MLP，到 MoE，再到工程里的局部 trick，都是在同一条结构线上做取舍。

3. 你看完之后能做什么
- 能画出 block 图
- 能解释组件在 block 中的位置
- 能把教科书结构和真实源码对应起来

## 1. 先从 token 进入 block

token 进入模型后，先变成 hidden state，再进入第一个规范化层。

- 你要看的是输入维度是否保持一致
- 你要记的是 residual 让主干信号可以跨层传播

## 2. 先走 Attention 支路

Attention 支路负责把当前位置和上下文联系起来。

- `04` 里看清 Q / K / V 的 head 关系
- `03` 里看清 RoPE 为什么作用在 Q / K 上
- `05` 里看清 Attention 如何和 residual 连接

这一步的关键不是“Attention 很重要”，而是理解它为什么既吃算力，也吃访存。

## 3. 再走 MLP / SwiGLU 支路

Attention 之后，token 会进入 MLP 路径。

- `02` 负责 gate / up / down 的分支直觉
- `01` 负责归一化前后的数值稳定性
- `05` 负责把它们重新放回同一个 block

这里最常见的误区，是把 MLP 当成“纯线性层堆叠”，忽略门控结构。

## 4. 理解 MoE 是怎么替换 dense MLP 的

如果结构升级到 MoE，那么 dense MLP 会被 router + experts 替换。

- `06` 看 token 如何被分配
- `07` 看为什么要做负载均衡

MoE 的本质不是多几个专家，而是把容量和路由显式拆开。

## 5. 回到真实模型实现

`08` 负责把结构从“教科书 block”拉回真实模型。

- 有些实现会改 norm 位置
- 有些实现会做权重共享
- 有些实现会对命名和局部顺序做工程化调整

这一步的目标，是让你看到源码时不会被命名变化误导。

## 可视化建议

这一页最适合配一张连续流图，直接画出 token hidden state 的路径：

```text
token embedding
   -> RMSNorm
   -> Attention
   -> Residual
   -> RMSNorm
   -> SwiGLU / MLP
   -> Residual
   -> optional MoE / tricks
```

如果要再进一步，可以在图的旁边标出 `01-08` 的对应位置，这样学习者就能把“单节内容”重新放回“整块结构”。

## 阅读建议

1. 先看 `05`，建立 block 总图。
2. 再回到 `01 / 02 / 03 / 04` 补每个支路。
3. 如果关心稀疏化，再看 `06 / 07`。
4. 最后看 `08`，确认你学到的是“结构规则”，不是某一份实现的细节。

## 和其他专题的连接

- 接到 [训练微调闭环专题](../fine_tuning_training/intro.md) 时，重点是理解 LoRA 会触达哪些层。
- 接到 [推理优化专题](../inference_optimization/intro.md) 时，重点是理解 Attention 和 block 为什么会影响 TTFT / TPOT。
- 接到 [显存优化与性能调优专题](../memory_performance_tuning/intro.md) 时，重点是理解 residual、attention、MLP 和 MoE 的资源占用差异。

## 样板说明

这页的作用，是给其他横向专题提供“连续讲述”的样板。后续其他专题如果要补厚，可以优先照着这三层来写：

- 一条主故事线
- 一组可直接引用的文献锚点
- 一张能把核心结构讲明白的图
