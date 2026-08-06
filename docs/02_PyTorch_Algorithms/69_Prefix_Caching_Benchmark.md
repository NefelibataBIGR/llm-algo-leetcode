# 69. Prefix Caching Benchmark | 前缀缓存基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `Caching`, `Benchmark` | **目标人群：** 推理缓存与 serving 工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/69_Prefix_Caching_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前缀缓存和 chunked prefill 的价值在于减少重复计算，但它们是否真的划算，取决于命中率、复用率和额外管理开销。本节把前缀缓存收成一个项目页：先定义缓存问题和 workload，再比较收益与开销，最后输出部署判断。

## 前置阅读

**导语：** 先看 vLLM、RadixAttention、分块预填充和解码策略，再做前缀缓存基准；这页重点是缓存命中口径和部署边界。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_VLLM_PagedAttention.md)
- [23. Speculative Decoding | 推测解码](./23_Speculative_Decoding.md)
- [24. SGLang RadixAttention | SGLang RadixAttention](./24_SGLang_RadixAttention.md)
- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](./34_Prefix_Caching_and_Chunked_Prefill.md)

### Step 1: 定义前缀缓存问题
先回答一个问题：这次 benchmark 要比较的是缓存命中率、重复计算减少量，还是端到端延迟收益？

- 固定 prompt 分布、请求重用模式、batch size 和 max context length。
- 明确 candidate 的缓存策略、chunk size 和 eviction 口径。
- 统一记录命中率、重复 token 比例、TTFT、吞吐和缓存管理开销。
- 先确定 workload，再判断前缀缓存是否值得接入。

#### 图解：22-24-34 如何收束到 69 前缀缓存基准

`69` 把缓存机制从单点技巧收成一个可部署的 benchmark。

```text
22 vLLM             paged KV and serving behavior
      │
24 RadixAttention   prefix reuse and route sharing
      │
34 Prefix cache     chunked prefill and reuse policy
      │
      ▼
69 Cache bench     hit rate + reuse gain + management overhead
```

项目页最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成前缀缓存命中率、收益和项目判断
# 目标：把缓存收益转成可比较的 benchmark 报告

def summarize_prefix_cache(runs):
    # ==========================================
    # TODO 1: 汇总缓存指标
    # 提示：统计命中率、重复 token 比例、TTFT 和吞吐。
    # ==========================================
    return {
        'run_count': 0,
        'avg_hit_rate': 0.0,
        'avg_ttft_ms': 0.0,
        'avg_throughput': 0.0,
    }

def compare_cache_policy(baseline, candidate):
    # ==========================================
    # TODO 2: 比较缓存策略
    # 提示：对比命中率、TTFT 改善和缓存管理开销。
    # ==========================================
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'hit_rate_delta': 0.0,
        'ttft_delta_ms': 0.0,
        'overhead_delta': 0.0,
    }

def should_enable_prefix_cache(candidate, min_hit_rate):
    # ==========================================
    # TODO 3: 判断是否启用缓存
    # 提示：命中率太低时不要继续推进。
    # ==========================================
    return {
        'enable_cache': False,
        'min_hit_rate': min_hit_rate,
    }

```


```python
# 测试你的实现
def test_prefix_cache_benchmark_template():
    try:
        baseline = {'name': 'baseline', 'hit_rate': 0.0, 'ttft_ms': 130, 'overhead': 0.0}
        candidate = {'name': 'prefix_cache', 'hit_rate': 0.65, 'ttft_ms': 100, 'overhead': 0.15}
        runs = [
            {'hit_rate': 0.6, 'ttft_ms': 110, 'throughput': 120},
            {'hit_rate': 0.7, 'ttft_ms': 95, 'throughput': 140},
        ]
        summary = summarize_prefix_cache(runs)
        assert 'run_count' in summary, '缓存汇总字段缺失！'
        comp = compare_cache_policy(baseline, candidate)
        assert 'hit_rate_delta' in comp and 'ttft_delta_ms' in comp, '缓存对比字段不完整！'
        decision = should_enable_prefix_cache(candidate, min_hit_rate=0.6)
        assert 'enable_cache' in decision, '缓存判断字段缺失！'
        print('测试通过：前缀缓存 benchmark 模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_prefix_cache_benchmark_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总缓存指标
def summarize_prefix_cache(runs):
    run_count = len(runs)
    if run_count == 0:
        return {
            'run_count': 0,
            'avg_hit_rate': 0.0,
            'avg_ttft_ms': 0.0,
            'avg_throughput': 0.0,
        }

    avg_hit_rate = sum(run.get('hit_rate', 0.0) for run in runs) / run_count
    avg_ttft_ms = sum(run.get('ttft_ms', 0.0) for run in runs) / run_count
    avg_throughput = sum(run.get('throughput', 0.0) for run in runs) / run_count
    return {
        'run_count': run_count,
        'avg_hit_rate': avg_hit_rate,
        'avg_ttft_ms': avg_ttft_ms,
        'avg_throughput': avg_throughput,
    }

# TODO 2: 比较缓存策略
def compare_cache_policy(baseline, candidate):
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'hit_rate_delta': candidate.get('hit_rate', 0.0) - baseline.get('hit_rate', 0.0),
        'ttft_delta_ms': candidate.get('ttft_ms', 0.0) - baseline.get('ttft_ms', 0.0),
        'overhead_delta': candidate.get('overhead', 0.0) - baseline.get('overhead', 0.0),
    }

# TODO 3: 判断是否启用缓存
def should_enable_prefix_cache(candidate, min_hit_rate):
    return {
        'enable_cache': candidate.get('hit_rate', 0.0) >= min_hit_rate,
        'min_hit_rate': min_hit_rate,
    }

```

### 解析

**1. TODO 1: 汇总缓存指标**
- **实现方式**：统计缓存命中率、TTFT 和吞吐的平均值，形成基线。
- **关键点**：缓存机制的收益高度依赖 workload 结构，必须先把输入分布说清楚。
- **项目意义**：这一步把零散请求场景转成可比较的 benchmark 结果。

**2. TODO 2: 比较缓存策略**
- **实现方式**：计算 candidate 相对 baseline 的命中率、延迟和管理开销差值。
- **关键点**：前缀缓存不是白捡收益，管理开销可能抵消一部分收益。
- **项目意义**：这一步帮助判断缓存策略是否真适合当前 serving 场景。

**3. TODO 3: 判断是否启用缓存**
- **实现方式**：用最小命中率门槛做启用判断。
- **关键点**：阈值判断要先于部署，否则 benchmark 结论无法落地。
- **项目意义**：把实验结果收束成是否启用缓存的工程决策。
