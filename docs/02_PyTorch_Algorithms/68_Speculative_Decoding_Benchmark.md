# 68. Speculative Decoding Benchmark | 推测解码基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `Inference`, `Benchmark` | **目标人群：** 推理优化与系统评估

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/68_Speculative_Decoding_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

推测解码的价值不在“能不能接入”，而在同一 workload 下是否真的能降低首 token 延迟、改善吞吐，并且不会把验证复杂度抬到不可接受的水平。本节把推测解码收成一个项目页：先固定 benchmark workload，再比较 baseline 与 candidate，最后输出是否继续推进的判断。

## 前置阅读

**导语：** 先看解码策略、FlashAttention、vLLM 和前缀缓存，再做推测解码基准；这页重点是指标口径和系统代价。
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [23. Speculative Decoding | 推测解码](./23_Speculative_Decoding.md)
- [24. SGLang RadixAttention | SGLang RadixAttention](./24_SGLang_RadixAttention.md)

### Step 1: 定义 benchmark workload
先回答一个问题：这次 benchmark 要比较的是 TTFT、TPOT、吞吐，还是在固定质量约束下的整体收益？

- 固定模型、prompt 分布、batch size、max new tokens 和解码温度。
- 明确 candidate 的 draft 模型、验证策略和接受率口径。
- 统一记录 TTFT、TPOT、吞吐、acceptance rate 和额外开销。
- 先确定 benchmark 场景，再讨论推测解码是否值得接入。

#### 图解：20-24 如何收束到 68 推测解码基准

`68` 把推测解码从算法概念收成一个统一的系统 benchmark。

```text
20 FlashAttention   attention compute baseline
      │
21 Decoding         TTFT / TPOT / sampling path
      │
22 vLLM             paged KV and serving behavior
      │
23 Spec Decode      draft + verify acceptance flow
      │
24 RadixAttention   prefix reuse and serving cache
      │
      ▼
68 Spec bench      workload + metric + cost + decision
```

项目页最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成推测解码 workload、指标比较和项目判断
# 目标：把推测解码结果整理成可比较的 benchmark 报告

def summarize_speculative_benchmark(runs):
    # ==========================================
    # TODO 1: 汇总 benchmark 指标
    # 提示：统计 TTFT、TPOT、吞吐、acceptance rate 和平均延迟。
    # ==========================================
    return {
        'run_count': 0,
        'avg_ttft_ms': 0.0,
        'avg_tpot_ms': 0.0,
        'avg_throughput': 0.0,
        'avg_acceptance_rate': 0.0,
    }

def compare_speculative_vs_baseline(baseline, candidate):
    # ==========================================
    # TODO 2: 比较 baseline 与 candidate
    # 提示：对比 TTFT、TPOT、吞吐和额外验证开销。
    # ==========================================
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'ttft_delta_ms': 0.0,
        'throughput_delta': 0.0,
        'acceptance_delta': 0.0,
    }

def should_deploy_speculative(candidate, min_acceptance_rate):
    # ==========================================
    # TODO 3: 判断是否值得推进
    # 提示：acceptance rate 低于门槛时直接判定不适合继续。
    # ==========================================
    return {
        'deployable': False,
        'min_acceptance_rate': min_acceptance_rate,
    }

```


```python
# 测试你的实现
def test_speculative_benchmark_template():
    try:
        baseline = {'name': 'baseline', 'ttft_ms': 120, 'throughput': 100, 'acceptance_rate': 0.0}
        candidate = {'name': 'spec_decode', 'ttft_ms': 90, 'throughput': 140, 'acceptance_rate': 0.72}
        runs = [
            {'ttft_ms': 120, 'tpot_ms': 18, 'throughput': 100, 'acceptance_rate': 0.0},
            {'ttft_ms': 90, 'tpot_ms': 12, 'throughput': 140, 'acceptance_rate': 0.72},
        ]
        summary = summarize_speculative_benchmark(runs)
        assert 'run_count' in summary, 'benchmark 汇总字段缺失！'
        comp = compare_speculative_vs_baseline(baseline, candidate)
        assert 'ttft_delta_ms' in comp and 'throughput_delta' in comp, '对比字段不完整！'
        decision = should_deploy_speculative(candidate, min_acceptance_rate=0.7)
        assert 'deployable' in decision, '推进判断字段缺失！'
        print('测试通过：推测解码 benchmark 模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_speculative_benchmark_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总 benchmark 指标
def summarize_speculative_benchmark(runs):
    run_count = len(runs)
    if run_count == 0:
        return {
            'run_count': 0,
            'avg_ttft_ms': 0.0,
            'avg_tpot_ms': 0.0,
            'avg_throughput': 0.0,
            'avg_acceptance_rate': 0.0,
        }

    avg_ttft_ms = sum(run.get('ttft_ms', 0.0) for run in runs) / run_count
    avg_tpot_ms = sum(run.get('tpot_ms', 0.0) for run in runs) / run_count
    avg_throughput = sum(run.get('throughput', 0.0) for run in runs) / run_count
    avg_acceptance_rate = sum(run.get('acceptance_rate', 0.0) for run in runs) / run_count
    return {
        'run_count': run_count,
        'avg_ttft_ms': avg_ttft_ms,
        'avg_tpot_ms': avg_tpot_ms,
        'avg_throughput': avg_throughput,
        'avg_acceptance_rate': avg_acceptance_rate,
    }

# TODO 2: 比较 baseline 与 candidate
def compare_speculative_vs_baseline(baseline, candidate):
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'throughput_delta': candidate.get('throughput', 0.0) - baseline.get('throughput', 0.0),
        'acceptance_delta': candidate.get('acceptance_rate', 0.0) - baseline.get('acceptance_rate', 0.0),
    }

# TODO 3: 判断是否值得推进
def should_deploy_speculative(candidate, min_acceptance_rate):
    return {
        'deployable': candidate.get('acceptance_rate', 0.0) >= min_acceptance_rate,
        'min_acceptance_rate': min_acceptance_rate,
    }

```

### 解析

**1. TODO 1: 汇总 benchmark 指标**
- **实现方式**：对多次运行的 TTFT、TPOT、吞吐和接受率求均值，得到基准结果。
- **关键点**：推测解码的收益必须放到统一 workload 上比较，否则 TTFT 和吞吐没有可比性。
- **项目意义**：把实验结果转成一张可直接讨论是否部署的指标表。

**2. TODO 2: 比较 baseline 与 candidate**
- **实现方式**：计算 candidate 相对 baseline 的延迟、吞吐和接受率变化。
- **关键点**：推测解码的代价常常藏在验证开销和接受率里，不能只看平均速度。
- **项目意义**：这一步帮助判断系统改动是净收益还是局部优化。

**3. TODO 3: 判断是否值得推进**
- **实现方式**：用接受率门槛决定 candidate 是否继续进入部署或进一步调优。
- **关键点**：阈值必须先定，否则 benchmark 没有工程决策意义。
- **项目意义**：把结果收束到“继续做还是停”的项目判断。
