# 02. Autograd 与 Attention Backward | Autograd and Attention Backward

## 页面目标

这一页把 PyTorch 的 autograd 机制和 attention 的反向链路放在一起看，目标是把“机制接口”和“算子级梯度路径”对齐。

## 核心问题

### 1. `grad_fn` 是什么

它表示当前张量是怎么被计算出来的，以及后续 backward 应该从哪条路径回传。

### 2. `saved_tensors` 为什么存在

因为 backward 往往需要前向中间状态。框架会把必要张量保留下来，以便反向时重用。

### 3. 自定义 `autograd.Function` 在做什么

它把 forward 和 backward 显式拆开，让你能手写梯度路径，验证自己是否真的理解了反传。

### 4. attention backward 的反向顺序怎么记

最实用的记法是：

`dV -> dP -> dS -> dQ / dK`

先求最容易的分支，再穿过 softmax 回到打分矩阵，最后回到 query 和 key。

## 机制分解

attention backward 里最容易混淆的，不是公式本身，而是中间状态的依赖顺序：

- `V` 的梯度最直接，因为它只受 attention 权重影响
- `P` 是 softmax 之后的概率矩阵，反向时要先穿过它
- `S` 是打分矩阵，通常还要经过缩放和 mask
- `Q / K` 依赖于打分矩阵对输入投影的链路

所以这条链路的重点是：

- 先找最容易求的梯度
- 再穿过 softmax 的耦合关系
- 最后回到 query 和 key 的投影路径

## 典型误区

- `grad_fn` 不是梯度本身，它只是记录这个张量的生成路径。
- `saved_tensors` 不是白送的，保存越多，显存压力越大。
- attention backward 的代价不只在公式，还在中间状态和 softmax 的稳定性处理。

## 对应来源

- `17 Autograd Basics`

## 经典论文

| 文献 | 读它的理由 |
|:---|:---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | attention 结构和 causal mask 的共同起点。 |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 看 attention 的 backward 代价如何被重新组织成更省 IO 的执行路径。 |
| [Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767) | 把 attention backward 放回 autodiff 的统一语境里看。 |

## 工程资料

| 资料 | 读它的理由 |
|:---|:---|
| [torch.autograd](https://docs.pytorch.org/docs/stable/autograd) | 直接看 PyTorch autograd 的官方定义、接口和行为边界。 |
| [torch.nn.attention](https://docs.pytorch.org/docs/stable/nn.attention.html) | 看 PyTorch 当前对 attention backend 的抽象和选择方式。 |

## 阅读建议

- 先把 `dV -> dP -> dS -> dQ / dK` 这条链背顺。
- 再回头看 `grad_fn` 和 `saved_tensors`。
- 如果你关心工程层面，顺手看 `FlashAttention`。
