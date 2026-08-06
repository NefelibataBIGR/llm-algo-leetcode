# 61. Model Architecture Exploration | 架构验证

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `Architecture`, `Model Design` | **目标人群：** 模型结构探索与训练工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/61_Model_Architecture_Exploration.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的章节已经把 Attention、LLaMA3 Block、Architecture Tricks、SFT 和端到端训练闭环拆开讲过，但真实项目里更常见的问题不是“某个模块怎么写”，而是“这个结构改动值不值得做”。本节把架构验证收成一个项目页：先定义基线和候选结构，再统一参数量、吞吐、loss 和显存口径，最后给出是否继续扩展的判断。

这页默认你已经知道常见 block 的组成方式，重点放在架构变体的对照、代价和项目决策。

## 前置阅读

**导语：** 先把注意力、block 结构、工程技巧和训练闭环看完，再做架构验证；这页的目标不是重讲基础，而是把结构差异转成可比较的实验结论。
- [04. Attention MHA/GQA | 注意力机制](./04_Attention_MHA_GQA.md)
- [05. LLaMA3 Block Tutorial | LLaMA3 Block 教程](./05_LLaMA3_Block_Tutorial.md)
- [08. Architecture Tricks | 架构技巧](./08_Architecture_Tricks.md)
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)

### Step 1: 定义架构探索目标
先回答一个问题：这次架构改动到底要优化什么，是参数量、吞吐、显存，还是最终 loss 和收敛速度？

- 固定 baseline 结构、训练数据、batch size、seq len、优化器和训练步数。
- 明确候选结构只改哪些模块，例如 attention、norm、FFN、残差路径或 block 组合。
- 统一记录参数量、step time、peak memory、train loss、val loss 和推理稳定性。
- 先设定预算边界，再判断候选结构是否值得进入后续验证。

#### 图解：04-13 如何收束到 61 架构验证

`61` 不重复实现基础模块，而是把前面几节已经讲过的组件收成一份可比较的结构验证报告。

```text
04 Attention      attention pattern / head grouping / masking
      │
05 Block          norm / attention / FFN / residual wiring
      │
08 Tricks         architecture-level efficiency constraints
      │
09 SFT            input_ids / labels / loss mask consistency
      │
13 E2E report      train loss / val loss / step time / memory
      │
      ▼
61 Architecture   baseline vs candidate + parameter ledger + decision
```

项目页最小产物：


```python
from typing import Any, Dict, Iterable, List

```


```python
# TODO: 完成架构候选摘要、差异对比、预算判断和项目结论
# 目标：把结构变体转成可比较的实验报告

def summarize_architecture_candidates(candidates, baseline_params):
    # ==========================================
    # TODO 1: 汇总候选架构
    # 提示：统计候选数、参数变化、候选名称和变化模块。
    # ==========================================
    # candidate_count = ???
    # best_candidate = ???
    # param_deltas = ???
    return {
        'candidate_count': 0,
        'baseline_params': baseline_params,
        'best_candidate': None,
        'param_deltas': {},
    }

def compare_architecture_pair(baseline, candidate):
    # ==========================================
    # TODO 2: 比较 baseline 和 candidate
    # 提示：对比 changed_modules、param_delta、memory_delta 和 score。
    # ==========================================
    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'changed_modules': [],
        'param_delta': 0,
        'memory_delta_mb': 0,
    }

def recommend_candidate(candidates, param_budget):
    # ==========================================
    # TODO 3: 给出推荐结论
    # 提示：在预算内选择综合 score 最优的候选。
    # ==========================================
    return {
        'recommended_name': None,
        'within_budget': False,
        'param_budget': param_budget,
    }

```


```python
# 测试你的实现
def test_architecture_project_template():
    try:
        baseline = {'name': 'baseline', 'params': 100, 'memory_mb': 1200}
        candidates = [
            {'name': 'small_norm', 'params': 96, 'memory_mb': 1100, 'changed_modules': ['norm'], 'score': 0.72},
            {'name': 'wide_ffn', 'params': 108, 'memory_mb': 1320, 'changed_modules': ['ffn'], 'score': 0.68},
        ]
        summary = summarize_architecture_candidates(candidates, baseline_params=baseline['params'])
        assert summary['candidate_count'] == 0 or summary['candidate_count'] == 2, '候选数统计不正确！'
        pair = compare_architecture_pair(baseline, candidates[0])
        assert 'baseline_name' in pair and 'candidate_name' in pair, '对比结果缺少必要字段！'
        decision = recommend_candidate(candidates, param_budget=102)
        assert 'recommended_name' in decision and 'within_budget' in decision, '推荐结果字段不完整！'
        print('测试通过：架构项目模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_architecture_project_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总候选架构
def summarize_architecture_candidates(candidates, baseline_params):
    candidate_count = len(candidates)
    param_deltas = {}
    best_candidate = None
    best_score = None

    for candidate in candidates:
        name = candidate.get('name', 'candidate')
        params = candidate.get('params', baseline_params)
        param_deltas[name] = params - baseline_params
        score = candidate.get('score')
        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_candidate = name

    return {
        'candidate_count': candidate_count,
        'baseline_params': baseline_params,
        'best_candidate': best_candidate,
        'param_deltas': param_deltas,
    }

# TODO 2: 比较 baseline 和 candidate
def compare_architecture_pair(baseline, candidate):
    baseline_params = baseline.get('params', 0)
    candidate_params = candidate.get('params', 0)
    baseline_memory = baseline.get('memory_mb', 0)
    candidate_memory = candidate.get('memory_mb', 0)

    return {
        'baseline_name': baseline.get('name', 'baseline'),
        'candidate_name': candidate.get('name', 'candidate'),
        'changed_modules': list(candidate.get('changed_modules', [])),
        'param_delta': candidate_params - baseline_params,
        'memory_delta_mb': candidate_memory - baseline_memory,
    }

# TODO 3: 给出推荐结论
def recommend_candidate(candidates, param_budget):
    feasible = [c for c in candidates if c.get('params', 10**9) <= param_budget]
    if not feasible:
        return {
            'recommended_name': None,
            'within_budget': False,
            'param_budget': param_budget,
        }

    best = min(feasible, key=lambda item: item.get('score', float('inf')))
    return {
        'recommended_name': best.get('name', 'candidate'),
        'within_budget': True,
        'param_budget': param_budget,
    }

```

### 解析

**1. TODO 1: 汇总候选架构**
- **实现方式**：遍历候选结构，记录候选数量、相对 baseline 的参数变化，以及当前评分最优的候选。
- **关键点**：架构探索必须先有统一口径，否则“更快”或“更省”无法和参数变化一起解释。
- **项目意义**：把结构差异收敛成一张能直接讨论取舍的实验表。

**2. TODO 2: 比较 baseline 和 candidate**
- **实现方式**：把 baseline 和 candidate 的参数量、显存、变更模块抽出来做差分。
- **关键点**：只看最终 loss 不够，架构改动的代价要跟工程指标一起看。
- **项目意义**：帮助判断候选是否只是“换形状”，还是确实带来可解释收益。

**3. TODO 3: 给出推荐结论**
- **实现方式**：在预算内选择综合 score 最优的候选；没有候选满足预算时直接返回不可用。
- **关键点**：项目页的结论必须和预算绑定，而不是只报一个分数。
- **项目意义**：这一步把实验结果转成下一轮训练或产品接入的决策。
