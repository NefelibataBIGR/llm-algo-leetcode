# 42. Activation Offload | 激活卸载

**难度：** Hard | **环境：** CPU-first

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/42_Activation_Offload.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

**标签：** `显存优化`, `激活值`, `Offload` | **目标人群：** 显存优化学习者

---

## 本节导读

当 activation 太大、GPU 显存放不下时，除了 checkpointing 以外，还有一条路线：把一部分“暂时不热”的激活临时搬到 CPU 或 host memory，等反向传播需要时再取回来。本节不做真实的分布式搬运实现，而是用一个最小 offload 计划器把“省了多少显存、付出多少搬运成本、值不值得”这条链路跑通。

这是一节**机制原理节**：它和 `19` 是兄弟关系。`19` 主讲 checkpointing 的重算路线；`42` 主讲 offload 的搬运路线。两者都在回答“怎么把训练显存压下来”，但实现机制不同，代价模型也不同。

在显存路线里，一个实用判断可以先保持简单：如果 GPU 显存是硬约束、而额外重算会明显拖慢训练，可以优先评估 offload；如果 host / PCIe / NVLink 带宽本身已经紧张，或者搬运时间很可能吞掉收益，就不要急着把 offload 当成默认答案，而应先回到 checkpointing、batch 或 mixed precision 这些更便宜的手段。

**关键词：** `offload`, `transfer`, `bandwidth`

---

## 前置阅读

**导语：** 先看 checkpointing 和反向传播，再看 offload 会更容易把“重算”与“搬运”分开。

- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [18. Activation and Loss Backward | 激活与损失反向](./18_Activation_and_Loss_Backward.md)
- [P0: 07. CPU/GPU Heterogeneous Scheduling | CPU/GPU 异构调度](../01_Hardware_Math_and_Systems/07_CPU_GPU_Heterogeneous_Scheduling.md)

## 相关阅读

**导语：** 学完 offload 后，下一步重点是把搬运策略放回性能分析和对比验证里，确认显存收益是否真的值得它带来的时延代价。

- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)
- [76. Activation Checkpoint Offload Benchmark | Checkpoint 与 Offload 对比项目](./76_Activation_Checkpoint_Offload_Benchmark.md)

---
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

```


```python
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
    汇总一次 activation offload 计划。
    """
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

    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (c.keep_score, c.bytes_, c.name),
    )

    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]

    # ==========================================
    # TODO 1: 汇总 offload 结果和显存/带宽指标
    # 提示：先算 offloaded_bytes = total_bytes - kept_bytes，
    # 再根据带宽估算 transfer_ms，并补出 pressure_ratio / saved_ratio。
    # ==========================================
    # offloaded_bytes = ???
    # transfer_ms = ???
    # pressure_ratio = ???
    # saved_ratio = ???

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
    根据 offload summary 返回 'accept' / 'tune' / 'reject'。
    """
    if summary.offloaded_bytes <= 0:
        return "reject"

    # ==========================================
    # TODO 2: 补全策略判断逻辑
    # 提示：先判断什么时候可以 accept，
    # 再判断“有收益但搬运偏贵”的 tune，剩下的统一 reject。
    # ==========================================
    # if ???:
    #     return "accept"
    # if ???:
    #     return "tune"

    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    """
    用“单位时间省下的显存”做一个最小对比，看看 offload 和 checkpointing 哪条更划算。
    """
    # ==========================================
    # TODO 3: 完成 offload 与 checkpointing 的性价比比较
    # 提示：两边都按“节省字节数 / 额外时间”算 score，
    # 再比较哪个更大，返回 preferred。
    # ==========================================
    # offload_score = ???
    # checkpoint_score = ???
    # preferred = ???
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }

```

### 测试

运行下面的测试，检查你的 offload 计划、策略判断和路线比较是否正确。

```python
def test_activation_offload():
    try:
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
    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了类型错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了参数或数值错误")
        else:
            print(f"代码可能未完成，导致了断言失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


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
    汇总一次 activation offload 计划。
    """
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

    candidates = sorted(
        [c for c in normalized if c.offloadable],
        key=lambda c: (c.keep_score, c.bytes_, c.name),
    )

    for chunk in candidates:
        if kept_bytes <= gpu_budget_bytes:
            break
        kept_bytes -= chunk.bytes_
        offloaded_names.append(chunk.name)

    kept_names = [c.name for c in normalized if c.name not in offloaded_names]

    # ==========================================
    # TODO 1: 汇总 offload 结果和显存/带宽指标
    # 提示：先算 offloaded_bytes = total_bytes - kept_bytes，
    # 再根据带宽估算 transfer_ms，并补出 pressure_ratio / saved_ratio。
    # ==========================================
    offloaded_bytes = total_bytes - kept_bytes
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
    根据 offload summary 返回 'accept' / 'tune' / 'reject'。
    """
    if summary.offloaded_bytes <= 0:
        return "reject"

    # ==========================================
    # TODO 2: 补全策略判断逻辑
    # 提示：先判断什么时候可以 accept，
    # 再判断“有收益但搬运偏贵”的 tune，剩下的统一 reject。
    # ==========================================
    if summary.kept_bytes <= summary.gpu_budget_bytes and summary.saved_ratio >= min_saved_ratio and summary.transfer_ms <= max_transfer_ms:
        return "accept"
    if summary.saved_ratio >= min_saved_ratio / 2 and summary.transfer_ms <= max_transfer_ms * 2:
        return "tune"

    return "reject"


def compare_offload_vs_checkpointing(offload_summary: ActivationOffloadSummary, checkpoint_saved_bytes, checkpoint_extra_ms):
    """
    用“单位时间省下的显存”做一个最小对比，看看 offload 和 checkpointing 哪条更划算。
    """
    # ==========================================
    # TODO 3: 完成 offload 与 checkpointing 的性价比比较
    # 提示：两边都按“节省字节数 / 额外时间”算 score，
    # 再比较哪个更大，返回 preferred。
    # ==========================================
    offload_score = offload_summary.offloaded_bytes / max(offload_summary.transfer_ms, 1e-6)
    checkpoint_score = checkpoint_saved_bytes / max(checkpoint_extra_ms, 1e-6)
    preferred = "offload" if offload_score >= checkpoint_score else "checkpointing"
    return {
        "offload_score": round(offload_score, 3),
        "checkpoint_score": round(checkpoint_score, 3),
        "preferred": preferred,
    }
```

### 解析

**1. TODO 1：汇总 offload 结果和显存/带宽指标**
- **实现方式**：先用 `offloaded_bytes = total_bytes - kept_bytes` 得到真实搬运量，再按 `带宽 = bytes / time` 反推 `transfer_ms`，最后补出 `pressure_ratio` 和 `saved_ratio`。
- **关键点**：`kept_bytes` 反映 offload 后还留在 GPU 上的激活量，`offloaded_bytes` 反映这次计划实际搬走了多少。
- **工程意义**：这一组指标把“省了多少显存”和“付出了多少搬运代价”放到同一张账上，是后面做策略判断的前提。

**2. TODO 2：补全策略判断逻辑**
- **实现方式**：先排除“根本没有发生 offload”的情况，再按“收益足够且搬运可接受”判断 `accept`，最后把“有收益但还不够稳”的情况归到 `tune`。
- **关键点**：这里不是只看显存节省，也不是只看搬运时间，而是两者一起看。
- **工程意义**：offload 不是默认值得做的优化；只有当显存确实被压进预算，且传输成本没有吞掉收益时，才值得接受。

**3. TODO 3：完成 offload 与 checkpointing 的性价比比较**
- **实现方式**：两条路线都按“节省字节数 / 额外时间”计算一个最小 score，再比较谁更大。
- **关键点**：`offload_score` 近似表示“单位搬运时间换回多少显存”，`checkpoint_score` 近似表示“单位重算时间换回多少显存”。
- **工程意义**：这不是精确性能模型，而是一个最小决策框架，帮助你判断当前瓶颈更像带宽问题还是重算问题。