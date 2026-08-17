# 80. MoE Expert Parallel Benchmark | MoE 专家并行基准
**难度：** Hard | **环境：** CPU-first | **标签：** `并行通信`, `MoE`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/80_MoE_Expert_Parallel_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

MoE 专家并行如果只停在“吞吐提高了”这一层，很容易忽略更关键的问题：通信代价是否变大、负载是否失衡、训练稳定性是否被破坏。真正需要回答的不是“专家并行能不能跑”，而是“它在当前预算和路由配置下值不值得 adopt”。

本节的核心矛盾是吞吐收益、通信代价与负载稳定性之间的权衡：专家并行可以提升扩展能力和训练吞吐，也可能因为 all-to-all 通信、热点 expert 或路由抖动把收益抵消掉。做完这一节，你应该能输出一份 baseline vs expert parallel 的 benchmark 结论，而不只是记录几组吞吐数字。

因此，这一页把 MoE 专家并行收成一个最小项目交付入口：先定义 benchmark 目标，再确认 baseline 与通信口径合法，用统一口径比较吞吐、负载不均和通信代价，并把结论收成 `accept / tune / reject` 的项目判断。它直接承接 `06 / 07 / 47 / 79` 的 MoE 与并行直觉，并继续通向 `81` 的分布式推理逻辑验证和 `86` 的在线对齐 benchmark。

**关键词：** `MoE`, `expert parallel`, `communication`, `imbalance`, `delivery`

---
## 前置阅读

**导语：** 先把 MoE 路由、负载均衡、专家并行机制和基础并行 benchmark 理顺，再进入这个项目；本节默认你已经知道 expert routing 的基本对象，重点转向专家并行是否值得保留。
- [06. MoE Router | MoE 路由](./06_MoE_Router.md)
- [07. MoE Load Balancing Loss | MoE 负载均衡损失](./07_MoE_Load_Balancing_Loss.md)
- [47. MoE Expert Parallel | MoE 专家并行](./47_MoE_Expert_Parallel.md)
- [79. Distributed Parallel Benchmark | 分布式并行基准](./79_Distributed_Parallel_Benchmark.md)

## 相关阅读

**导语：** 做完 MoE 专家并行 benchmark 后，最自然的下一步是把并行结论推进到分布式推理验证，或回看对齐场景下的系统收益。
- [81. Distributed Inference Logic Validation | 分布式推理逻辑验证](./81_Distributed_Inference_Project.md)
- [86. DPO Online Benchmark | DPO 在线基准](./86_DPO_Online_Benchmark.md)
---
### Step 1: 定义 expert parallel benchmark 目标

- 固定 expert 数、router 策略、batch size、seq len 和训练步数。
- 明确 baseline 是无 expert parallel 还是其他并行策略。
- 统一记录通信时间、负载不均、吞吐和训练稳定性。

### Step 2: baseline 和通信口径先要合法

- 专家并行 benchmark 不能脱离 baseline 并行口径单独比较。
- 如果 baseline 的通信统计或吞吐波动本身不稳定，候选结果就没有解释空间。
- 至少要先确认 baseline 的通信时间、吞吐和负载分布是可复现的。

### Step 3: 用统一口径比较收益与代价

- 专家并行项目必须同时看吞吐、通信代价、负载不均和训练稳定性，不能只挑单项吞吐收益下结论。
- 但副作用会体现在 all-to-all 通信、负载不均和训练噪声上。
- 如果吞吐变好，但负载不均明显恶化，候选通常只能进入 `tune`，而不是直接 `accept`。

### Step 4: 输出 benchmark 结论

- 专家并行最终不是输出“吞吐有没有涨”，而是输出这套 expert parallel 配置在当前预算下是否值得继续保留、微调或放弃。
- 最终结论建议统一为 `accept / tune / reject`。
- 若进入 `tune`，下一轮优先回调 router、capacity factor、expert 分组或通信拓扑，而不是盲目继续加 expert。
#### 图解：06-07-46-47-79 如何收束到 80 MoE 基准

`80` 把 MoE 的路由、均衡、通信和并行基线收成一个项目页。

```text
06 Router         expert routing / token dispatch
      │
07 Load balance   auxiliary balancing pressure
      │
46 NCCL           communication hotspot evidence
      │
47 Expert parallel dispatch / gather and communication cost
      │
79 Parallel bench baseline distributed throughput
      │
      ▼
80 MoE bench      throughput + imbalance + communication + delivery decision
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| baseline | 吞吐、通信时间、负载分布 | 保证比较合法 |
| candidate | expert 配置、通信变化、负载变化 | 解释收益来源 |
| 对比 | throughput gain、comm delta、imbalance delta | 判断是否值得 adopt |
| 决策 | accept / tune / reject | 输出 benchmark 结论 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：workload 汇总、baseline 对比、项目判断
# 目标：把通信与负载均衡结果整理成 benchmark 报告

def summarize_moe_parallel_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def compare_moe_parallel_to_baseline(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    raise NotImplementedError("请先完成 TODO 代码！")

def recommend_moe_parallel_run(
    baseline: dict[str, float],
    candidate: dict[str, float],
    max_imbalance: float,
    min_stability: float,
) -> dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_moe_parallel_benchmark_template():
    baseline = {'name': 'baseline', 'comm_ms': 40, 'imbalance': 0.25, 'throughput': 100, 'stability': 0.82}
    candidate = {'name': 'expert_parallel', 'comm_ms': 48, 'imbalance': 0.12, 'throughput': 128, 'stability': 0.80}
    summary = summarize_moe_parallel_runs([baseline, candidate])
    assert summary['run_count'] == 2
    assert summary['best_throughput_run'] == 'expert_parallel'
    comparison = compare_moe_parallel_to_baseline(baseline, candidate)
    assert comparison['comm_delta_ms'] == 8
    assert comparison['imbalance_delta'] == -0.13
    assert comparison['throughput_gain'] == 28
    assert comparison['stability_delta'] == -0.02
    decision = recommend_moe_parallel_run(baseline, candidate, max_imbalance=0.15, min_stability=0.78)
    assert decision['decision'] == 'accept'
    assert decision['next_action'] == 'promote_to_cluster_eval'


test_moe_parallel_benchmark_template()
print('测试通过：MoE 专家并行基准模板可以工作。')

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
def summarize_moe_parallel_runs(runs: list[dict[str, float]]) -> dict[str, object]:
    best = max(runs, key=lambda item: item.get('throughput', 0.0))
    avg_imbalance = sum(item.get('imbalance', 0.0) for item in runs) / len(runs) if runs else 0.0
    return {'run_count': len(runs), 'best_throughput_run': best.get('name', 'run'), 'avg_imbalance': avg_imbalance}


def compare_moe_parallel_to_baseline(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        'comm_delta_ms': candidate.get('comm_ms', 0.0) - baseline.get('comm_ms', 0.0),
        'imbalance_delta': round(candidate.get('imbalance', 0.0) - baseline.get('imbalance', 0.0), 4),
        'throughput_gain': candidate.get('throughput', 0.0) - baseline.get('throughput', 0.0),
        'stability_delta': round(candidate.get('stability', 0.0) - baseline.get('stability', 0.0), 4),
    }


def recommend_moe_parallel_run(
    baseline: dict[str, float],
    candidate: dict[str, float],
    max_imbalance: float,
    min_stability: float,
) -> dict[str, object]:
    comparison = compare_moe_parallel_to_baseline(baseline, candidate)
    if (
        comparison['throughput_gain'] > 0
        and candidate.get('imbalance', 10**9) <= max_imbalance
        and candidate.get('stability', -10**9) >= min_stability
    ):
        return {'decision': 'accept', 'reason': '吞吐收益、负载均衡和稳定性都达标', 'next_action': 'promote_to_cluster_eval'}
    if comparison['throughput_gain'] > 0 and candidate.get('stability', -10**9) >= min_stability:
        return {'decision': 'tune', 'reason': '吞吐收益可用，但负载不均或通信代价仍偏高', 'next_action': 'refine_router_or_capacity'}
    return {'decision': 'reject', 'reason': '收益不足或训练稳定性不达标', 'next_action': 'fallback_to_parallel_baseline'}

```

### 解析

这页现在按 `measure -> compare -> decide` 的最小 MoE 专家并行项目闭环组织，不再只是单独比较吞吐和通信代价。

#### TODO 1

- 实现方式：先汇总 run 数量和平均负载不均衡，再找出吞吐最高的 run。
- 关键点：`best_throughput_run` 只是帮助定位最值得回看的 candidate，不等于最终项目结论。
- 项目意义：先把 workload 摘要做平，后面才能在同一 expert 并行设置下比较收益与代价。

#### TODO 2

- 实现方式：统一计算 `comm_delta_ms`、`imbalance_delta`、`throughput_gain` 和 `stability_delta`。
- 关键点：吞吐和稳定性越高越好，通信代价和负载不均衡越低越好，所以指标方向必须统一。
- 项目意义：这一步把 MoE 专家并行从“技巧演示”转成“吞吐、通信和稳定性能否一起成立”的 benchmark 对比。

#### TODO 3

- 实现方式：先复用 baseline 对比结果，再按吞吐收益、imbalance 边界和稳定性输出 `accept / tune / reject`。
- 关键点：`tune` 主要对应吞吐收益已出现，但 router、capacity factor 或通信拓扑还没有一起收稳。
- 项目意义：MoE 专家并行项目最后要回答的是“这套并行方案值不值得继续扩到真实集群”，而不是只看某个吞吐数字。