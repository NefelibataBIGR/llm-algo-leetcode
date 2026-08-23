# 性能分析（Profiling）正文

这页只做 profiling 问题的判断框架：不重复 `intro` 的路线入口，也不写 `walkthrough` 的连续故事。

## 使用顺序

先用第一张表判断问题类别，再用第二张表检查证据是否足够，最后把结果交给 `05 -> 06`。如果还不能回答“改善了什么、付出了什么代价、是否可复现”，就停留在取证阶段，不进入优化决策。

## 工具选择表

工具选择服从“先便宜、后深入”的原则。不要因为已经安装了 profiler，就跳过问题定义和基线测量。

| 观察目标 | 首选工具 | 输出证据 | 需要升级工具的条件 |
|:---|:---|:---|:---|
| 总耗时和吞吐 | `time.perf_counter()` + 固定 workload | step time、吞吐、延迟 | 需要区分 CPU / GPU 阶段时进入 `torch.profiler` |
| GPU kernel 时间 | `torch.cuda.Event` | CUDA elapsed time | 需要查看调用链或阶段关系时导出 trace |
| 算子热点和阶段拆分 | `torch.profiler` | `key_averages`、CPU/CUDA 时间、Chrome Trace | 需要观察 stream、搬运和系统重叠时进入 Nsight Systems |
| 显存峰值和驻留 | `torch.cuda.max_memory_allocated()`、`max_memory_reserved()` | peak / reserved memory | 需要解释分配时序或碎片时使用 memory timeline / snapshot |
| 多卡等待和通信重叠 | NCCL trace / debug log、`torch.profiler` | collective 时间、等待和 overlap | 需要系统级关键路径时进入 Nsight Systems |
| kernel 访存和硬件利用率 | Nsight Compute | occupancy、带宽、Tensor Core 等 | 仅当瓶颈已定位到具体 kernel 时使用 |

工具输出必须回到同一份 benchmark 记录，至少保留 workload、环境、基线、优化策略、指标和结论；孤立的 trace 截图不能作为优化成功的证据。

## 最小实践模板

下面三段模板对应最常见的三类证据。先完成轻量计时，再按问题选择 trace 或显存观察；不要在没有基线的情况下直接比较 profiler 输出。

### 1. GPU 计时：使用 CUDA Event

```python
import torch

def measure_cuda_ms(fn, warmup=3, iters=10):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA Event timing")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters
```

`CUDA Event` 用于测量 GPU 执行时间；如果测量的是完整请求，还要另外记录端到端延迟，不能用 kernel 时间替代服务延迟。

### 2. 算子与阶段热点：使用 `torch.profiler`

```python
import torch
from torch.profiler import profile, ProfilerActivity

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

with profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    for _ in range(3):
        run_one_step()  # 替换为实际的 forward / train step

print(prof.key_averages().table(
    sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
    row_limit=15,
))
prof.export_chrome_trace("benchmarks/results/profile_trace.json")
```

先看 `key_averages` 判断热点，再打开 Chrome Trace 看阶段关系、同步点和数据搬运；不要只凭一行算子耗时判断端到端收益。

### 3. 显存证据：峰值、reserved 和快照

```python
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    run_one_step()
    torch.cuda.synchronize()

    memory = {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
    }
    print(memory)
    snapshot = torch.cuda.memory_snapshot()
```

峰值指标用于判断是否装得下；`memory_snapshot()` 用于进一步观察 allocator 状态。若要解释 activation、checkpoint 或 offload 的对象变化，应把这类证据带入显存专题的 `76 -> 75`，而不是在 Profiling 专题直接做预算决策。

基础来源可回看 [Part 00: 17 Profiling Basics](../../00_Prerequisites/17_PyTorch_Profiling_Basics.ipynb)、[Part 00: 20 Profiling and Memory Ledger](../../00_Prerequisites/20_Profiling_and_Memory_Ledger.ipynb) 和 [Part 01: 13 Profiling and Bottleneck Analysis](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)。

## 判断表

先分清问题在时间热点、memory timeline、通信等待还是 benchmark 验证，再判断证据链是否足够支撑下一步动作。

| 现象 | 优先判断 | 先看哪条线 | 常见动作 |
|:---|:---|:---|:---|
| 系统变慢了，但不知道慢在哪 | `time hotspot` | [02](./02_time_breakdown_and_trace_reading.md) | 先看 operator、trace、阶段拆分 |
| 时间和显存一起波动 | `memory residency` | [03](./03_memory_timeline_and_residency.md) | 看 allocation、residency、timeline |
| 多卡收益不稳 | `wait / overlap mismatch` | [04](./04_communication_wait_and_overlap.md) | 看同步等待、communication trace |
| before / after 结果说不清 | `benchmark gap` | [05](./05_benchmark_design_and_regression_validation.md), [06](./06_diagnosis_and_action_decision.md) | 回到 workload 和回归验证 |

| 检查项 | 主要回答什么 | 常见误判 |
|:---|:---|:---|
| 时间热点 | 慢点是不是已经被定位清楚 | 看一眼 trace 就下结论 |
| memory timeline | 时间问题是不是伴随显存驻留问题 | 看显存高就直接去省显存 |
| communication wait | 多卡是不是在等而不是在算 | 多卡慢就一定是算子问题 |
| benchmark | 这次优化是不是可复现、可比较 | before / after 口径不统一 |

## 本节要点

这页的职责不是教工具按钮，而是把 profiling 里最常见的判断点压成一张表。路线入口留给 `intro`，连续故事留给 `walkthrough`。

## 最小决策模板

记录 `现象 -> 假设 -> 证据 -> before / after -> 动作` 五个字段。缺少其中任一字段时，结论应标记为“待验证”，而不是直接写成“优化成功”。
