# 47. MoE Expert Parallel | MoE 专家并行

**难度：** Hard | **环境：** CPU-first | **标签：** `MoE`, `并行`, `通信`, `专家路由` | **目标人群：** 大模型并行与系统优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/47_MoE_Expert_Parallel.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

MoE 不是简单地“把参数变多”，它真正难的地方在于：token 被路由到不同专家后，专家往往分布在不同设备上，训练和推理都会引入额外通信、负载不均和 capacity 溢出。

这一节不做工业级并行实现，而是用纯 Python / PyTorch 风格的最小模拟，把 expert placement、token dispatch、capacity limit 和 all-to-all 代价串起来。学完后，你应该能看清“MoE 专家并行为什么是并行问题，而不只是模型结构问题”。
## 前置阅读

**导语：** 先看最小必要前置，再进入本节。
- [06. MoE Router | MoE 路由器](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [29. Tensor Parallelism Sim | Tensor Parallelism 模拟](./29_Tensor_Parallelism_Sim.md)
- [46. Communication Profiling with NCCL | NCCL 通信 Profiling](./46_Communication_Profiling_with_NCCL.md)
## 相关阅读

- [2.8](./2_8.md)：先看分布式并行策略组页的整体位置。
- [2.9](./2_9.md)：看并行能力如何进入项目实战与 benchmark。
### Step 1: 核心思想与痛点

MoE 的核心是“稀疏激活”：每个 token 只激活少量专家，所以理论上可以在较低计算代价下扩展参数规模。

但一旦专家分布在不同设备上，就会出现三个典型问题：
- token 必须被 dispatch 到对应设备上的专家；
- 不同专家收到的 token 数量可能差很多；
- 所有 token 计算完后，还要把结果 gather 回来。

所以 MoE 专家并行的核心不是“让更多专家跑起来”，而是“让专家分布、token 路由和通信代价同时可控”。
### Step 2: 专家并行的实现框架

这一节先把最小结构拆清楚：

- `routes`：每个 token 命中的专家列表。
- `expert_loads`：每个 expert 实际收到多少路由。
- `capacity`：每个 expert 在给定 `capacity_factor` 下允许处理多少路由。
- `overflow`：超过 capacity 的路由数，代表溢出或丢弃风险。
- `dispatch_bytes` / `all_to_all_bytes`：近似估算通信代价。

这不是完整的并行系统，但足够用来判断 MoE expert parallel 的瓶颈在负载还是通信。
#### 图解：token 如何进入专家并行

```text
token batch
   -> router 选 top-k experts
   -> dispatch 到各 device 上的 experts
   -> expert 计算
   -> gather 回原始 batch
```

如果某个 expert 拿到的 token 明显更多，就会出现热点；如果 dispatch 很频繁，就会被 all-to-all 通信拖慢。
### Step 3: 代价模型与边界

设 `A` 表示总路由次数，也就是 `sum(len(route_i))`。

- 平均负载：`A / num_experts`
- capacity：`ceil((A / num_experts) * capacity_factor)`
- overflow：`sum(max(load_e - capacity, 0))`
- dispatch 近似开销：`A * hidden_size * bytes_per_elem`
- all-to-all 近似开销：`2 * dispatch_bytes`

其中 `capacity_factor` 越大，越不容易溢出，但也会让每个 expert 预留更多空间；`hidden_size` 和 dtype 字节数越大，通信代价越重。
### Step 4: 动手实战

完成下面三个函数：

1. `summarize_expert_parallel`：汇总 expert 负载、capacity、overflow 和通信量。
2. `recommend_moe_parallel_plan`：根据负载与 overflow 给出 `accept / tune / reject`。
3. `compare_dense_vs_moe_cost`：给一个最小 dense vs MoE 的成本对比。

注意：这里要做的是**并行与通信的最小评估**，不是训练项目。
### 提示

- `routes` 里的每个元素表示一个 token 被路由到哪些专家。
- `capacity = ceil((total_routes / num_experts) * capacity_factor)`。
- `overflow` 是超出 capacity 的路由数，不要把它和 `total_routes` 混在一起。
- `balance_ratio = max(expert_loads) / avg_load`，它反映最重的 expert 是否明显偏热。
- 这里不要模拟真实分布式 kernel，只要做最小的并行与通信账本即可。

```python
from collections import Counter
from dataclasses import dataclass
import math

@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    dispatch_bytes: int
    all_to_all_bytes: int
    balance_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    """
    TODO:
    1. 统计每个 expert 的负载 expert_loads。
    2. 按 capacity_factor 计算 capacity。
    3. 统计 overflow、dispatch_bytes、all_to_all_bytes 和 balance_ratio。
    4. 返回 MoEParallelSummary。
    """
    # TODO 1: 校验 num_experts 是否为正
    # TODO 2: 遍历 routes，统计每个 expert 的路由数
    # TODO 3: 计算 total_tokens、total_routes 和 expert_loads
    # TODO 4: 计算 avg_load、capacity、overflow
    # TODO 5: 估算通信量 dispatch_bytes / all_to_all_bytes
    # TODO 6: 计算 balance_ratio
    raise NotImplementedError


def recommend_moe_parallel_plan(summary: MoEParallelSummary) -> str:
    """
    TODO:
    根据 summary 的 overflow 和 balance_ratio，返回 'accept' / 'tune' / 'reject'。
    """
    # TODO 1: 空 summary 直接 reject
    # TODO 2: 无 overflow 且负载均匀 -> accept
    # TODO 3: 少量 overflow 且负载尚可 -> tune
    # TODO 4: 其他情况 -> reject
    raise NotImplementedError


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    """
    TODO:
    返回一个最小 dense vs MoE 成本对比字典。
    """
    # TODO 1: 计算 dense_compute_units
    # TODO 2: 计算 moe_routes
    # TODO 3: 估算 moe_dispatch_bytes / moe_all_to_all_bytes
    # TODO 4: 返回 sparsity_ratio
    raise NotImplementedError

```

### 测试

运行下面的测试，检查你的并行负载统计、容量判断和代价估算是否正确。

```python
def test_moe_expert_parallel():
    balanced = summarize_expert_parallel(
        routes=[[0, 1], [1, 2], [0, 2]],
        num_experts=3,
        hidden_size=4,
        bytes_per_elem=2,
        capacity_factor=1.0,
    )
    assert balanced.total_tokens == 3
    assert balanced.total_routes == 6
    assert balanced.expert_loads == [2, 2, 2]
    assert balanced.capacity == 2
    assert balanced.overflow == 0
    assert balanced.dispatch_bytes == 48
    assert balanced.all_to_all_bytes == 96
    assert balanced.balance_ratio == 1.0
    assert recommend_moe_parallel_plan(balanced) == "accept"

    tuned = summarize_expert_parallel(
        routes=[[0], [0], [0], [1]],
        num_experts=2,
        hidden_size=8,
        bytes_per_elem=2,
        capacity_factor=1.0,
    )
    assert tuned.total_routes == 4
    assert tuned.expert_loads == [3, 1]
    assert tuned.capacity == 2
    assert tuned.overflow == 1
    assert recommend_moe_parallel_plan(tuned) == "tune"

    rejected = summarize_expert_parallel(
        routes=[[0]] * 10 + [[1]],
        num_experts=2,
        hidden_size=8,
        bytes_per_elem=2,
        capacity_factor=1.0,
    )
    assert rejected.total_routes == 11
    assert rejected.expert_loads == [10, 1]
    assert rejected.capacity == 6
    assert rejected.overflow == 4
    assert rejected.balance_ratio > 1.8
    assert recommend_moe_parallel_plan(rejected) == "reject"

    empty = summarize_expert_parallel(
        routes=[],
        num_experts=2,
        hidden_size=8,
        bytes_per_elem=2,
        capacity_factor=1.0,
    )
    assert empty.total_tokens == 0
    assert empty.total_routes == 0
    assert empty.expert_loads == [0, 0]
    assert empty.capacity == 0
    assert empty.overflow == 0
    assert recommend_moe_parallel_plan(empty) == "reject"

    cost = compare_dense_vs_moe_cost(token_count=4, top_k=2, hidden_size=8, num_experts=4)
    assert cost["dense_compute_units"] == 32
    assert cost["moe_routes"] == 8
    assert cost["moe_dispatch_bytes"] == 128
    assert cost["moe_all_to_all_bytes"] == 256
    assert cost["sparsity_ratio"] == 0.5
    print("moe expert parallel passed")


test_moe_expert_parallel()
```

---

🛑 **STOP HERE** 🛑
<br><br><br><br><br><br><br><br><br><br>
> 请先尝试自己完成代码并跑通测试。<br>
> 如果你正在 Colab 中运行，并且遇到困难没有思路，可以向下滚动查看参考答案。
<br><br><br><br><br><br><br><br><br><br>

---
## 参考代码与解析

### 代码

```python
from collections import Counter
from dataclasses import dataclass
import math

@dataclass
class MoEParallelSummary:
    total_tokens: int
    total_routes: int
    expert_loads: list
    capacity: int
    overflow: int
    dispatch_bytes: int
    all_to_all_bytes: int
    balance_ratio: float


def summarize_expert_parallel(routes, num_experts, hidden_size, bytes_per_elem=2, capacity_factor=1.0):
    # TODO 1: 校验 num_experts 是否为正
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    # TODO 2: 遍历 routes，统计每个 expert 的路由数
    counts = Counter()
    total_tokens = len(routes)
    total_routes = 0
    for token_routes in routes:
        for expert_id in token_routes:
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError("expert id out of range")
            counts[expert_id] += 1
            total_routes += 1

    # TODO 3: 计算 expert_loads
    expert_loads = [counts.get(i, 0) for i in range(num_experts)]

    # TODO 4: 计算 avg_load、capacity、overflow
    avg_load = total_routes / num_experts if num_experts else 0.0
    capacity = math.ceil(avg_load * capacity_factor) if total_routes else 0
    overflow = sum(max(load - capacity, 0) for load in expert_loads)

    # TODO 5: 估算通信量 dispatch_bytes / all_to_all_bytes
    dispatch_bytes = total_routes * hidden_size * bytes_per_elem
    all_to_all_bytes = dispatch_bytes * 2

    # TODO 6: 计算 balance_ratio
    balance_ratio = (max(expert_loads) / avg_load) if avg_load > 0 else 0.0

    return MoEParallelSummary(
        total_tokens=total_tokens,
        total_routes=total_routes,
        expert_loads=expert_loads,
        capacity=capacity,
        overflow=overflow,
        dispatch_bytes=dispatch_bytes,
        all_to_all_bytes=all_to_all_bytes,
        balance_ratio=round(balance_ratio, 3),
    )


def recommend_moe_parallel_plan(summary: MoEParallelSummary) -> str:
    # TODO 1: 空 summary 直接 reject
    if summary.total_routes == 0:
        return "reject"
    # TODO 2: 无 overflow 且负载均匀 -> accept
    if summary.overflow == 0 and summary.balance_ratio <= 1.2:
        return "accept"
    # TODO 3: 少量 overflow 且负载尚可 -> tune
    if summary.overflow <= 1 and summary.balance_ratio <= 1.8:
        return "tune"
    # TODO 4: 其他情况 -> reject
    return "reject"


def compare_dense_vs_moe_cost(token_count, top_k, hidden_size, num_experts, bytes_per_elem=2):
    # TODO 1: 计算 dense_compute_units
    dense_compute_units = token_count * hidden_size
    # TODO 2: 计算 moe_routes
    moe_routes = token_count * top_k
    # TODO 3: 估算 moe_dispatch_bytes / moe_all_to_all_bytes
    moe_dispatch_bytes = moe_routes * hidden_size * bytes_per_elem
    moe_all_to_all_bytes = moe_dispatch_bytes * 2
    # TODO 4: 返回 sparsity_ratio
    return {
        "dense_compute_units": dense_compute_units,
        "moe_routes": moe_routes,
        "moe_dispatch_bytes": moe_dispatch_bytes,
        "moe_all_to_all_bytes": moe_all_to_all_bytes,
        "sparsity_ratio": round(top_k / num_experts, 3) if num_experts else 0.0,
    }

```

### 解析

**1. TODO 1-3：`summarize_expert_parallel` 的三个核心步骤**
- 先校验 `num_experts`，避免非法输入。
- 再遍历 `routes` 统计每个 expert 的负载。
- 最后把负载汇总成 capacity、overflow、通信量和 balance_ratio。

**2. TODO 4：`recommend_moe_parallel_plan` 的判断规则**
- `accept`：没有 overflow，且负载足够均匀。
- `tune`：有少量 overflow，但还在可接受范围。
- `reject`：负载太偏或 overflow 太多。

**3. TODO 5：`compare_dense_vs_moe_cost` 的作用**
- 它不做真实 kernel，而是给出最小的 dense vs MoE 代价对比。
- 这样你能快速判断 MoE 的稀疏激活是否值回通信开销。

**4. 这一页的边界**
- 它讲的是 MoE 专家并行和通信代价。
- 它不负责完整训练闭环，也不负责项目级 benchmark。