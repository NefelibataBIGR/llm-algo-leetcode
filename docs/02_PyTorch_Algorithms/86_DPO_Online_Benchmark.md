# 86. DPO Online Benchmark | DPO 在线基准

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `Alignment`, `Benchmark` | **目标人群：** 后训练与对齐工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/86_DPO_Online_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

在线 DPO 的问题不是有没有用，而是更新节奏、偏好噪声和稳定性代价是否能被控制在可接受范围内。本节把在线 DPO 收成一个项目页：先定义在线基准目标，再比较稳定性与更新成本，最后输出是否继续推进的判断。

## 前置阅读

**导语：** 先看 DPO、GRPO、偏好数据和对齐评估，再做在线 DPO 基准；这页重点是在线更新和稳定性边界。
- [44. Preference Data and Evaluation | 偏好数据与评估](./44_Preference_Data_and_Evaluation.md)
- [45. DPO Preference Project | DPO 偏好项目](./45_DPO_Preference_Project.md)
- [46. GRPO Groupwise Alignment Project | GRPO 分组对齐项目](./46_GRPO_Groupwise_Alignment_Project.md)
- [51. Online DPO | 在线 DPO](./51_Online_DPO.md)
- [84. DPO Preference Project | DPO 偏好项目](./84_DPO_Preference_Project.md)
- [85. GRPO Groupwise Alignment Project | GRPO 分组对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)

### Step 1: 定义在线基准目标
先回答一个问题：这次 benchmark 要比较的是偏好更新速度、稳定性，还是在线样本带来的收益？

- 固定初始模型、偏好流、更新频率、batch size 和评估窗口。
- 明确 candidate 的在线更新规则、采样策略和安全阈值。
- 统一记录偏好胜率、更新步时、波动幅度和训练稳定性。
- 先定义评估窗口，再判断在线 DPO 是否值得推进。

#### 图解：44-46-51-84-85 如何收束到 86 在线基准

`86` 把偏好数据、离线对齐和在线更新组合成一张 benchmark 表。

```text
44 Preference      preference data and evaluation baseline
      │
45 DPO project     offline preference alignment
      │
46 GRPO project    groupwise alignment baseline
      │
51 Online DPO      online update rule and sampling stream
      │
84 DPO project     project-level preference tuning
      │
85 GRPO project    project-level groupwise alignment
      │
      ▼
86 Online bench   update speed + stability + preference gain
```

项目页最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成在线 DPO 更新、稳定性比较和项目判断
# 目标：把在线偏好更新整理成 benchmark 报告

def summarize_online_dpo_runs(runs):
    # ==========================================
    # TODO 1: 汇总在线 benchmark
    # 提示：统计胜率、更新耗时、波动幅度和稳定性指标。
    # ==========================================
    return {
        'run_count': 0,
        'avg_win_rate': 0.0,
        'avg_update_ms': 0.0,
        'avg_stability': 0.0,
    }

def compare_online_dpo(baseline, candidate):
    # ==========================================
    # TODO 2: 比较 baseline 与 candidate
    # 提示：对比胜率、更新成本和稳定性变化。
    # ==========================================
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'win_rate_delta': 0.0,
        'update_ms_delta': 0.0,
        'stability_delta': 0.0,
    }

def should_continue_online_dpo(candidate, min_win_rate):
    # ==========================================
    # TODO 3: 判断是否继续推进
    # 提示：胜率不足或稳定性太差时直接停止。
    # ==========================================
    return {
        'continue_training': False,
        'min_win_rate': min_win_rate,
    }

```


```python
# 测试你的实现
def test_online_dpo_benchmark_template():
    try:
        baseline = {'name': 'baseline', 'win_rate': 0.52, 'update_ms': 90, 'stability': 0.8}
        candidate = {'name': 'online_dpo', 'win_rate': 0.58, 'update_ms': 110, 'stability': 0.76}
        runs = [
            {'win_rate': 0.55, 'update_ms': 100, 'stability': 0.78},
            {'win_rate': 0.58, 'update_ms': 110, 'stability': 0.76},
        ]
        summary = summarize_online_dpo_runs(runs)
        assert 'run_count' in summary, '在线 DPO 汇总字段缺失！'
        comp = compare_online_dpo(baseline, candidate)
        assert 'win_rate_delta' in comp and 'stability_delta' in comp, '在线 DPO 对比字段不完整！'
        decision = should_continue_online_dpo(candidate, min_win_rate=0.56)
        assert 'continue_training' in decision, '在线 DPO 判断字段缺失！'
        print('测试通过：DPO 在线 benchmark 模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_online_dpo_benchmark_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总在线 benchmark
def summarize_online_dpo_runs(runs):
    run_count = len(runs)
    if run_count == 0:
        return {
            'run_count': 0,
            'avg_win_rate': 0.0,
            'avg_update_ms': 0.0,
            'avg_stability': 0.0,
        }

    avg_win_rate = sum(run.get('win_rate', 0.0) for run in runs) / run_count
    avg_update_ms = sum(run.get('update_ms', 0.0) for run in runs) / run_count
    avg_stability = sum(run.get('stability', 0.0) for run in runs) / run_count
    return {
        'run_count': run_count,
        'avg_win_rate': avg_win_rate,
        'avg_update_ms': avg_update_ms,
        'avg_stability': avg_stability,
    }

# TODO 2: 比较 baseline 与 candidate
def compare_online_dpo(baseline, candidate):
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'win_rate_delta': candidate.get('win_rate', 0.0) - baseline.get('win_rate', 0.0),
        'update_ms_delta': candidate.get('update_ms', 0.0) - baseline.get('update_ms', 0.0),
        'stability_delta': candidate.get('stability', 0.0) - baseline.get('stability', 0.0),
    }

# TODO 3: 判断是否继续推进
def should_continue_online_dpo(candidate, min_win_rate):
    return {
        'continue_training': candidate.get('win_rate', 0.0) >= min_win_rate and candidate.get('stability', 0.0) >= 0.75,
        'min_win_rate': min_win_rate,
    }

```

### 解析

**1. TODO 1: 汇总在线 benchmark**
- **实现方式**：统计 win rate、更新耗时和稳定性的平均值。
- **关键点**：在线 DPO 不是单次实验，必须看运行窗口内的稳定表现。
- **项目意义**：把在线更新结果收成一张可比较的指标表。

**2. TODO 2: 比较 baseline 与 candidate**
- **实现方式**：计算 candidate 相对 baseline 的胜率、更新成本和稳定性变化。
- **关键点**：在线方法的收益必须和训练成本一起看，不能只盯着 win rate。
- **项目意义**：帮助判断在线更新值不值得继续做。

**3. TODO 3: 判断是否继续推进**
- **实现方式**：用胜率和稳定性双门槛决定是否继续训练。
- **关键点**：在线方法的停止条件必须先定义，否则项目会无限追样本。
- **项目意义**：把实验结果转换成是否上线或继续调优的结论。
