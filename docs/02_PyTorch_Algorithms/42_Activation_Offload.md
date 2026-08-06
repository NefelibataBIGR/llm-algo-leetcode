# 42. Activation Offload | 激活卸载

**难度：** Hard | **环境：** CPU-first

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/42_Activation_Offload.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `激活显存优化`, `Offload`, `CPU/GPU Transfer` | **目标人群：** 显存优化与系统调优入门者

---

## 本节导读

当 activation 太大、GPU 显存放不下时，除了 checkpointing 以外，还有一条路线：把一部分“暂时不热”的激活临时搬到 CPU 或 host memory，等反向传播需要时再取回来。本节不做真实的分布式搬运实现，而是用一个最小 offload 计划器把“省了多少显存、付出多少搬运成本、值不值得”这条链路跑通。

这是一节**机制原理节**：它和 `19` 是兄弟关系。`19` 主讲 checkpointing 的重算路线；`47` 主讲 offload 的搬运路线。两者都在回答“怎么把训练显存压下来”，但实现机制不同，代价模型也不同。

**关键词：** `offload`, `transfer`, `bandwidth`

## 前置阅读

**导语：** 先看 checkpointing 和反向传播，再看 offload 会更容易把“重算”与“搬运”分开。

- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [18. Activation and Loss Backward | 激活与损失反向](./18_Activation_and_Loss_Backward.md)
- [P0: 07. CPU/GPU Heterogeneous Scheduling | CPU/GPU 异构调度](../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.md)

## 相关阅读

**导语：** 学完 offload 后，可以继续看训练性能分析和调优闭环，理解搬运成本是否真的换回了足够的收益。

- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [74. Profiling-Driven End-to-End Optimization | Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)

### Step 1: 核心思想与痛点

> **Activation Offload 的基本思路：**
> 当某些中间激活在短时间内不会被频繁访问，但又不能像 checkpointing 那样完全丢掉时，可以把它们临时搬到 CPU 或 host memory。这样 GPU 显存就能腾出空间，继续放更大的 batch、更长的序列或更多层的中间状态。
>
> **它和 checkpointing 的区别：**
> - checkpointing 是**重算**：前向时少存，反向时再算一遍。
> - offload 是**搬运**：前向时先搬走，反向时再搬回来。
>
> **核心权衡：**
> offload 省的是 GPU 显存，但代价是 PCIe / NVLink / 内存带宽上的搬运时间。网络越慢、搬得越多，收益就越容易被传输开销吞掉。

### Step 2: 代价模型与边界

设一组激活块的总大小为 `A`，GPU 可用预算为 `B`，带宽为 `bw`。

- 如果 `A <= B`，说明理论上不需要 offload。
- 如果 `A > B`，可以优先把“较冷”的激活块搬出 GPU。
- 搬运代价近似为：`transfer_ms = offloaded_bytes / bandwidth`
- offload 不是越多越好；如果搬运时间过长，整体 step time 会被拖慢。

这一步的重点不是精确模拟硬件，而是把“GPU 显存节省”和“数据搬运成本”放到同一张账上。

### Step 3: 代码实现框架

我们先定义激活块规格，再根据 `keep_score` 做一个最小的 offload 计划器：

1. 统计总激活大小。
2. 按 `keep_score` 从低到高优先 offload。
3. 计算 offload 后的 GPU 剩余占用和搬运时间。
4. 再根据收益和代价给出 `accept / tune / reject`。

### Step 4: 动手实战

完成下面三个函数：

1. `summarize_activation_offload`：汇总 offload 计划、显存节省和搬运成本。
2. `recommend_offload_policy`：根据节省比例和搬运时间给出 `accept / tune / reject`。
3. `compare_offload_vs_checkpointing`：比较 offload 和 checkpointing 的性价比，看看哪条路线更划算。

### 提示

- `keep_score` 越高，表示这块激活越应该留在 GPU 上。
- offload 先搬运“冷”的块，不要先搬运最重要的块。
- `bandwidth_gbps` 越低，搬运时间越长。
- 比较 offload 和 checkpointing 时，可以先看“单位时间省了多少显存”。


```python
from dataclasses import dataclass

@dataclass
class ActivationChunkSpec:
    name: str
    bytes_: int
    keep_score: float
    offloadable: bool = True

@dataclass
class ActivationOffloadSummary:
    total_bytes: int
    gpu_budget_bytes: int
    kept_bytes: int
    offloaded_bytes: int
    transfer_ms: float
    pressure_ratio: float
    saved_ratio: float
    offloaded_names: list
    kept_names: list


def summarize_activation_offload(chunks, gpu_budget_bytes, bandwidth_gbps=12.0):
    """
    TODO:
    1. 汇总总激活大小与 GPU budget。
    2. 按 keep_score 从低到高选择可 offload 的块。
    3. 计算 offloaded_bytes、kept_bytes、transfer_ms 和 saved_ratio。
    4. 返回 ActivationOffloadSummary。
    """
    # TODO 1: 校验参数
    if gpu_budget_bytes <= 0:
        raise ValueError("gpu_budget_bytes must be positive")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")

    normalized = []
    for c in chunks:
        if isinstance(c, ActivationChunkSpec):
            normalized.append(c)
        elif isinstance(c, dict):
            normalized.append(ActivationChunkSpec(**c))
        else:
            raise TypeError("chunks must contain ActivationChunkSpec or dict")

    total_bytes = sum(c.bytes_ for c in normalized)
    kept_bytes = total_bytes
    offloaded_names = []

    # TODO 2: 按 keep_score 构建 offload 顺序
    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (c.keep_score, c.bytes_, c.name),
    )

    # TODO 3: 计算 offload 选择与剩余显存
    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]
    offloaded_bytes = total_bytes - kept_bytes

    # TODO 4: 估算 transfer_ms / pressure_ratio / saved_ratio
    transfer_ms = offloaded_bytes / (bandwidth_gbps * (1024 ** 3)) * 1000 if offloaded_bytes else 0.0
    pressure_ratio = total_bytes / gpu_budget_bytes
    saved_ratio = offloaded_bytes / total_bytes if total_bytes else 0.0

    return ActivationOffloadSummary(
        total_bytes=total_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        kept_bytes=kept_bytes,
        offloaded_bytes=offloaded_bytes,
        transfer_ms=round(transfer_ms, 2),
        pressure_ratio=round(pressure_ratio, 3),
        saved_ratio=round(saved_ratio, 3),
        offloaded_names=offloaded_names,
        kept_names=kept_names,
    )


def recommend_offload_policy(summary: ActivationOffloadSummary, min_saved_ratio=0.25, max_transfer_ms=60.0):
    """
    TODO:
    根据 offload summary 返回 'accept' / 'tune' / 'reject'。
    """
    # TODO 1: 没有实际 offload 就 reject
    if summary.offloaded_bytes <= 0:
        return "reject"

    # TODO 2: 显存已经被压进预算且搬运时间可接受 -> accept
    if summary.kept_bytes <= summary.gpu_budget_bytes and summary.saved_ratio >= min_saved_ratio and summary.transfer_ms <= max_transfer_ms:
        return "accept"

    # TODO 3: 有收益但搬运成本偏高 -> tune
    if summary.saved_ratio >= min_saved_ratio / 2 and summary.transfer_ms <= max_transfer_ms * 2:
        return "tune"

    # TODO 4: 其他情况 -> reject
    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    """
    TODO:
    用“单位时间省下的显存”做一个最小对比，看看 offload 和 checkpointing 哪条更划算。
    """
    # TODO 1: 计算 offload_score
    offload_score = offload_summary.offloaded_bytes / max(offload_summary.transfer_ms, 1e-6)

    # TODO 2: 计算 checkpoint_score
    checkpoint_score = checkpoint_saved_bytes / max(checkpoint_extra_ms, 1e-6)

    # TODO 3: 返回 preferred 路线与分数
    preferred = "offload" if offload_score >= checkpoint_score else "checkpointing"
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }


chunks = [
    ActivationChunkSpec("embed", 256 * 1024 * 1024, 0.9),
    ActivationChunkSpec("mid_a", 192 * 1024 * 1024, 0.4),
    ActivationChunkSpec("mid_b", 160 * 1024 * 1024, 0.2),
    ActivationChunkSpec("tail", 128 * 1024 * 1024, 0.7),
]
summary = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=8.0)
print(summary)
print(recommend_offload_policy(summary))
print(compare_offload_vs_checkpointing(summary, checkpoint_saved_bytes=300 * 1024 * 1024, checkpoint_extra_ms=80.0))
```

### 测试

运行下面的测试，检查你的 offload 计划、策略判断和路线比较是否正确。

```python
def test_activation_offload():
    chunks = [
        ActivationChunkSpec("embed", 256 * 1024 * 1024, 0.9),
        ActivationChunkSpec("mid_a", 192 * 1024 * 1024, 0.4),
        ActivationChunkSpec("mid_b", 160 * 1024 * 1024, 0.2),
        ActivationChunkSpec("tail", 128 * 1024 * 1024, 0.7),
    ]
    summary = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=8.0)
    assert summary.total_bytes == 736 * 1024 * 1024
    assert summary.offloaded_bytes == 352 * 1024 * 1024
    assert summary.kept_bytes == 384 * 1024 * 1024
    assert summary.offloaded_names == ["mid_b", "mid_a"]
    assert summary.kept_names == ["embed", "tail"]
    assert 42.0 < summary.transfer_ms < 44.0
    assert 0.47 < summary.saved_ratio < 0.49
    assert recommend_offload_policy(summary) == "accept"

    tuned = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=4.0)
    assert recommend_offload_policy(tuned) == "tune"

    rejected = summarize_activation_offload(chunks, gpu_budget_bytes=384 * 1024 * 1024, bandwidth_gbps=1.0)
    assert recommend_offload_policy(rejected) == "reject"

    empty = summarize_activation_offload(chunks, gpu_budget_bytes=1024 * 1024 * 1024, bandwidth_gbps=8.0)
    assert empty.offloaded_bytes == 0
    assert recommend_offload_policy(empty) == "reject"

    better_offload = compare_offload_vs_checkpointing(summary, checkpoint_saved_bytes=300 * 1024 * 1024, checkpoint_extra_ms=80.0)
    assert better_offload["preferred"] == "offload"

    better_ckpt = compare_offload_vs_checkpointing(summary, checkpoint_saved_bytes=500 * 1024 * 1024, checkpoint_extra_ms=20.0)
    assert better_ckpt["preferred"] == "checkpointing"
    print("activation offload passed")


test_activation_offload()
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
from dataclasses import dataclass

@dataclass
class ActivationChunkSpec:
    name: str
    bytes_: int
    keep_score: float
    offloadable: bool = True

@dataclass
class ActivationOffloadSummary:
    total_bytes: int
    gpu_budget_bytes: int
    kept_bytes: int
    offloaded_bytes: int
    transfer_ms: float
    pressure_ratio: float
    saved_ratio: float
    offloaded_names: list
    kept_names: list


def summarize_activation_offload(chunks, gpu_budget_bytes, bandwidth_gbps=12.0):
    # TODO 1: 校验参数
    if gpu_budget_bytes <= 0:
        raise ValueError("gpu_budget_bytes must be positive")
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")

    normalized = []
    for c in chunks:
        if isinstance(c, ActivationChunkSpec):
            normalized.append(c)
        elif isinstance(c, dict):
            normalized.append(ActivationChunkSpec(**c))
        else:
            raise TypeError("chunks must contain ActivationChunkSpec or dict")

    total_bytes = sum(c.bytes_ for c in normalized)
    kept_bytes = total_bytes
    offloaded_names = []

    # TODO 2: 按 keep_score 构建 offload 顺序
    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (c.keep_score, c.bytes_, c.name),
    )

    # TODO 3: 计算 offload 选择与剩余显存
    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]
    offloaded_bytes = total_bytes - kept_bytes

    # TODO 4: 估算 transfer_ms / pressure_ratio / saved_ratio
    transfer_ms = offloaded_bytes / (bandwidth_gbps * (1024 ** 3)) * 1000 if offloaded_bytes else 0.0
    pressure_ratio = total_bytes / gpu_budget_bytes
    saved_ratio = offloaded_bytes / total_bytes if total_bytes else 0.0

    return ActivationOffloadSummary(
        total_bytes=total_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        kept_bytes=kept_bytes,
        offloaded_bytes=offloaded_bytes,
        transfer_ms=round(transfer_ms, 2),
        pressure_ratio=round(pressure_ratio, 3),
        saved_ratio=round(saved_ratio, 3),
        offloaded_names=offloaded_names,
        kept_names=kept_names,
    )


def recommend_offload_policy(summary: ActivationOffloadSummary, min_saved_ratio=0.25, max_transfer_ms=60.0):
    # TODO 1: 没有实际 offload 就 reject
    if summary.offloaded_bytes <= 0:
        return "reject"

    # TODO 2: 显存已经被压进预算且搬运时间可接受 -> accept
    if summary.kept_bytes <= summary.gpu_budget_bytes and summary.saved_ratio >= min_saved_ratio and summary.transfer_ms <= max_transfer_ms:
        return "accept"

    # TODO 3: 有收益但搬运成本偏高 -> tune
    if summary.saved_ratio >= min_saved_ratio / 2 and summary.transfer_ms <= max_transfer_ms * 2:
        return "tune"

    # TODO 4: 其他情况 -> reject
    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    # TODO 1: 计算 offload_score
    offload_score = offload_summary.offloaded_bytes / max(offload_summary.transfer_ms, 1e-6)

    # TODO 2: 计算 checkpoint_score
    checkpoint_score = checkpoint_saved_bytes / max(checkpoint_extra_ms, 1e-6)

    # TODO 3: 返回 preferred 路线与分数
    preferred = "offload" if offload_score >= checkpoint_score else "checkpointing"
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }
```

### 解析

**1. TODO 1-4：offload 先看能搬多少，再看搬运代价**
- `total_bytes` 和 `gpu_budget_bytes` 先回答“是否真的需要 offload”。
- `keep_score` 先决定哪些块更应该留在 GPU。
- `offloaded_bytes`、`kept_bytes` 和 `transfer_ms` 再把收益和代价放到同一张账上。

**2. TODO 1-4：policy 看的是收益是否值得**
- 如果根本没发生 offload，就直接 `reject`。
- 如果显存压进预算，且搬运时间可接受，就 `accept`。
- 如果有收益但搬运成本偏高，就 `tune`。
- 这和 checkpointing 的思路不同：checkpointing 是重算，offload 是搬运。

**3. TODO 1-3：与 checkpointing 的比较**
- `offload_score` 粗略表示单位搬运时间能省多少显存。
- `checkpoint_score` 粗略表示单位重算时间能省多少显存。
- 两者不是硬性孰优孰劣，而是要看当前瓶颈是 PCIe / host memory，还是算力重算更划算。