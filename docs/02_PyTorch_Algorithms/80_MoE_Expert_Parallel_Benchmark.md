# 80. MoE Expert Parallel Benchmark | MoE 专家并行基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `MoE`, `Parallelism` | **目标人群：** 分布式训练与通信优化

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

MoE 专家并行的核心不是“把 expert 切开”，而是切分以后通信、dispatch 和 load balance 是否还能保持在可接受范围内。本节把 MoE 专家并行收成一个项目页：先定义 benchmark 目标，再比较通信与负载均衡，最后输出是否继续推进的判断。

## 前置阅读

**导语：** 先看 MoE 路由、负载均衡损失、通信剖析和分布式并行，再做 MoE 专家并行基准；这页重点是通信和负载均衡代价。
- [06. MoE Router | MoE 路由](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [46. Communication Profiling with NCCL | NCCL 通信剖析](./46_Communication_Profiling_with_NCCL.md)
- [47. MoE Expert Parallel | MoE 专家并行](./47_MoE_Expert_Parallel.md)
- [79. Distributed Parallel Benchmark | 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)

### Step 1: 定义 benchmark 目标
先回答一个问题：这次基准要比较的是通信开销、负载均衡，还是总吞吐和收敛稳定性？

- 固定 expert 数、router 策略、batch size、seq len 和训练步数。
- 明确 baseline 是无 expert parallel 还是其他并行策略。
- 统一记录 dispatch 时间、all-to-all 开销、token imbalance、step time 和吞吐。
- 先设定通信预算，再判断 MoE 专家并行是否值得推进。

#### 图解：06-07-46-47-79 如何收束到 80 MoE 基准

`80` 把 MoE 的路由、均衡、通信和并行基线收成一个项目页。

```text
06 Router         expert routing / token dispatch
      │
07 Load balance   auxiliary loss / expert usage balance
      │
46 NCCL profile   communication breakdown
      │
47 Expert parallel split experts across workers
      │
79 Distributed    parallel baseline comparison
      │
      ▼
80 MoE bench     dispatch + communication + balance + decision
```

项目页最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成 MoE 专家并行统计、对比和项目判断
# 目标：把通信与负载均衡结果整理成 benchmark 报告

def summarize_moe_parallel_runs(runs):
    # ==========================================
    # TODO 1: 汇总 MoE benchmark
    # 提示：统计 dispatch time、all-to-all time、imbalance 和 throughput。
    # ==========================================
    return {
        'run_count': 0,
        'avg_dispatch_ms': 0.0,
        'avg_comm_ms': 0.0,
        'avg_imbalance': 0.0,
        'avg_throughput': 0.0,
    }

def compare_moe_parallel(baseline, candidate):
    # ==========================================
    # TODO 2: 比较 baseline 与 candidate
    # 提示：对比通信时间、负载均衡和吞吐变化。
    # ==========================================
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'comm_delta_ms': 0.0,
        'imbalance_delta': 0.0,
        'throughput_delta': 0.0,
    }

def should_scale_moe(candidate, max_comm_ms):
    # ==========================================
    # TODO 3: 判断是否继续扩展
    # 提示：通信代价过高时不要继续推规模。
    # ==========================================
    return {
        'scale_up': False,
        'max_comm_ms': max_comm_ms,
    }

```


```python
# 测试你的实现
def test_moe_parallel_benchmark_template():
    try:
        baseline = {'name': 'baseline', 'comm_ms': 40, 'imbalance': 0.25, 'throughput': 100}
        candidate = {'name': 'expert_parallel', 'comm_ms': 55, 'imbalance': 0.12, 'throughput': 135}
        runs = [
            {'dispatch_ms': 12, 'comm_ms': 40, 'imbalance': 0.25, 'throughput': 100},
            {'dispatch_ms': 10, 'comm_ms': 55, 'imbalance': 0.12, 'throughput': 135},
        ]
        summary = summarize_moe_parallel_runs(runs)
        assert 'run_count' in summary, 'MoE 汇总字段缺失！'
        comp = compare_moe_parallel(baseline, candidate)
        assert 'comm_delta_ms' in comp and 'throughput_delta' in comp, 'MoE 对比字段不完整！'
        decision = should_scale_moe(candidate, max_comm_ms=60)
        assert 'scale_up' in decision, 'MoE 判断字段缺失！'
        print('测试通过：MoE 专家并行 benchmark 模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_moe_parallel_benchmark_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总 MoE benchmark
def summarize_moe_parallel_runs(runs):
    run_count = len(runs)
    if run_count == 0:
        return {
            'run_count': 0,
            'avg_dispatch_ms': 0.0,
            'avg_comm_ms': 0.0,
            'avg_imbalance': 0.0,
            'avg_throughput': 0.0,
        }

    avg_dispatch_ms = sum(run.get('dispatch_ms', 0.0) for run in runs) / run_count
    avg_comm_ms = sum(run.get('comm_ms', 0.0) for run in runs) / run_count
    avg_imbalance = sum(run.get('imbalance', 0.0) for run in runs) / run_count
    avg_throughput = sum(run.get('throughput', 0.0) for run in runs) / run_count
    return {
        'run_count': run_count,
        'avg_dispatch_ms': avg_dispatch_ms,
        'avg_comm_ms': avg_comm_ms,
        'avg_imbalance': avg_imbalance,
        'avg_throughput': avg_throughput,
    }

# TODO 2: 比较 baseline 与 candidate
def compare_moe_parallel(baseline, candidate):
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'comm_delta_ms': candidate.get('comm_ms', 0.0) - baseline.get('comm_ms', 0.0),
        'imbalance_delta': candidate.get('imbalance', 0.0) - baseline.get('imbalance', 0.0),
        'throughput_delta': candidate.get('throughput', 0.0) - baseline.get('throughput', 0.0),
    }

# TODO 3: 判断是否继续扩展
def should_scale_moe(candidate, max_comm_ms):
    return {
        'scale_up': candidate.get('comm_ms', 0.0) <= max_comm_ms,
        'max_comm_ms': max_comm_ms,
    }

```

### 解析

**1. TODO 1: 汇总 MoE benchmark**
- **实现方式**：统计 dispatch、通信、负载不均衡和吞吐的平均值。
- **关键点**：MoE 的 benchmark 不能只看最终速度，通信和 imbalance 是最核心的代价。
- **项目意义**：把分布式细节收成一张可比较的工程表。

**2. TODO 2: 比较 baseline 与 candidate**
- **实现方式**：计算 candidate 相对 baseline 的通信、均衡和吞吐变化。
- **关键点**：如果通信代价升高太多，吞吐提升也未必值得。
- **项目意义**：帮助判断专家并行是不是在当前规模下可用。

**3. TODO 3: 判断是否继续扩展**
- **实现方式**：用最大通信时间门槛决定是否继续放大规模。
- **关键点**：扩展条件应该和系统预算绑定，而不是只看模型指标。
- **项目意义**：把 benchmark 结果转成是否扩容的项目结论。
