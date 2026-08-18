# 76. Activation Checkpoint Offload Benchmark | Activation / Checkpoint / Offload 对比项目

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `Checkpoint/Offload`, `基准对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一节对应的真实项目问题不是“activation checkpointing 和 offload 怎么实现”，而是“在既定预算、吞吐约束和质量下限下，checkpoint、offload 或二者组合哪一种最值得采用”。真实工程里，显存优化的难点不在知道这些技巧，而在判断它们是否真的比 baseline 更值，以及代价落在速度、复杂度还是稳定性上。

本节的核心矛盾是显存收益与系统代价之间的权衡：checkpoint 往往会增加重算成本，offload 会引入数据搬运开销，组合方案虽然更省显存，但也可能把 step time 拉得不可接受。做完这一节，你应该能输出一份 baseline vs checkpoint vs offload vs hybrid 的项目结论，而不只是比较几组显存数字。

因此，这一页把 activation / checkpoint / offload 的显存策略收成一个最小项目交付入口：先固定预算与质量边界，再统一比较显存、吞吐、step time 和质量约束，最后把结果收成 `accept / tune / reject` 的项目结论。它直接承接 `19 / 42 / 43 / 75` 的机制与预算内容，并继续通向 `77-78` 的后续显存扩展位。

**关键词：** `activation`, `checkpoint`, `offload`, `memory`, `benchmark`

---
## 前置阅读

**导语：** 先把激活检查点、offload、统一内存和显存预算项目理顺，再进入这个项目；本节默认你已经知道这些技巧各自怎么省显存，重点转向在同一预算口径下哪种方案更值。
- [19. Activation Checkpointing | 激活检查点](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [42. Activation Offload | 激活卸载](./42_Activation_Offload.md)
- [43. Unified Memory Management | 统一内存管理](./43_Unified_Memory_Management.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)

## 相关阅读

**导语：** 做完这页后，最自然的下一步是继续到 profiling 证据链复查瓶颈，或留给后续显存扩展位做更细的 memory timeline 项目。
- [74. Profiling Driven End-to-End Optimization | Profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [75. Memory Budget Compression Project | 显存预算压缩项目](./75_Memory_Budget_Compression_Project.md)

### Step 1: 定义显存策略对比目标
先回答一个问题：当前训练任务下，你比较这些方案是为了压进预算、保住吞吐，还是在二者之间找最稳的折中？

- 固定模型、数据、batch size、seq len、训练步数、评测指标和质量下限，保证后面的策略比较都在同一口径下进行。
- 明确候选方案：baseline、checkpoint、offload，以及必要时的 hybrid 组合。
- 把预算边界先写清楚：显存上限、最低可接受吞吐和最大允许的 val loss 退化。
- 这一步的目标不是立刻挑一个方案，而是先把“什么叫值得采用”定义清楚。

### Step 2: 先确认 baseline 和方案口径合法
显存策略对比项目必须先确认 baseline 和候选方案的比较口径一致，否则后面的结果没有解释力。

- 先记录 baseline 的 peak memory、step time、samples/s 和 val loss。
- 再确认候选方案只改显存策略，不要把优化器、数据和训练步数一起改掉。
- 对组合方案，还要单独说明 checkpoint 颗粒度和 offload 范围。
- 如果 baseline 本身不稳定，或者方案口径前后不一致，后面的对比结论都不可信。

### Step 3: 用统一口径比较收益与代价
显存策略对比不能只看哪组显存最低，还要把吞吐、step time 和质量约束一起算进去。

- 至少统一比较 peak memory、step time、samples/s 和 val loss。
- 如果某个方案显存收益很大，但吞吐掉得太多或质量越过阈值，它通常只能进入 `tune` 或 `reject`。
- 如果某个方案显著压低显存，同时速度和质量都还在交付边界内，就可以进入 `accept`。
- 这一步的目标是把显存收益、执行代价和训练风险收成一张可比较的方案表。

### Step 4: 输出显存策略项目结论
显存策略对比项目最终不是输出“哪个技巧最省显存”，而是输出当前预算下最值得继续采用的策略组合。

- 项目结论建议统一成 `accept / tune / reject`。
- 输出最小报告时，至少包含候选方案、核心指标差异、是否满足预算与质量、以及下一轮动作。
- 若进入 `tune`，下一轮优先回调 checkpoint 颗粒度、offload 范围或组合方式，而不是直接再叠更多技巧。

#### 图解：19 / 42 / 43 / 75 如何收束到 76 显存策略对比项目

`76` 不重复解释单个显存技巧，而是把前面几节的机制和预算口径收成一份方案对比报告。

```text
19 Checkpoint / offload   memory saving intuition
      │
42 Activation offload     transfer and runtime trade-off
      │
43 Unified memory         system-side memory coordination
      │
75 Budget compression     budget and quality boundary
      ▼
76 Activation / Checkpoint / Offload Benchmark
      ├─ strategy candidates
      ├─ baseline vs checkpoint vs offload vs hybrid
      ├─ quality floor review
      └─ accept / tune / reject
```

项目页最小产物：

| 产物 | 你至少要记录什么 | 作用 |
|:---|:---|:---|
| 候选方案 | baseline / checkpoint / offload / hybrid | 固定比较对象 |
| 预算边界 | 显存上限、吞吐下限、质量阈值 | 固定方案判断边界 |
| 结果对比 | peak memory、step time、samples/s、val loss | 统一看收益与代价 |
| 项目结论 | accept / tune / reject | 输出策略选择 |


```python
from typing import Dict, List

```


```python
# 3 个核心 TODO：预算检查、候选汇总、项目结论
# 目标：把 baseline / checkpoint / offload / hybrid 的方案比较收束成一份项目报告

def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def summarize_memory_strategy_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    raise NotImplementedError("请先完成 TODO 代码！")

```


```python
# 测试你的实现
def test_memory_strategy_project():
    try:
        budget = {'memory_cap_mb': 12000.0, 'min_samples_per_s': 6.0}
        quality_floor = {'max_val_loss': 1.15}
        check = validate_strategy_budget(budget, quality_floor)
        assert check['is_valid'] is True, '预算检查应通过'
        assert check['missing_keys'] == [], '完整预算不应缺字段'

        candidates = [
            {'name': 'baseline', 'peak_memory_mb': 18000.0, 'samples_per_s': 8.0, 'val_loss': 1.06},
            {'name': 'checkpoint', 'peak_memory_mb': 11800.0, 'samples_per_s': 6.6, 'val_loss': 1.08},
            {'name': 'offload', 'peak_memory_mb': 10500.0, 'samples_per_s': 5.2, 'val_loss': 1.09},
            {'name': 'hybrid', 'peak_memory_mb': 9800.0, 'samples_per_s': 6.1, 'val_loss': 1.11},
        ]
        summary = summarize_memory_strategy_candidates(candidates, budget, quality_floor)
        assert summary['feasible_count'] == 2, '应有两个方案满足预算与质量'
        assert summary['best_candidate'] == 'hybrid', 'hybrid 应成为最省显存的可行方案'

        decision = decide_memory_strategy_project(summary)
        assert decision['decision'] == 'accept', '可行且最优的方案应被接受'

        hard_summary = summarize_memory_strategy_candidates(
            [
                {'name': 'checkpoint', 'peak_memory_mb': 13000.0, 'samples_per_s': 6.4, 'val_loss': 1.10},
                {'name': 'offload', 'peak_memory_mb': 11000.0, 'samples_per_s': 4.8, 'val_loss': 1.12},
            ],
            budget,
            quality_floor,
        )
        hard_decision = decide_memory_strategy_project(hard_summary)
        assert hard_decision['decision'] == 'reject', '没有满足预算与质量时应 reject'
        print('所有测试通过！')
    except NotImplementedError:
        print('请先完成 TODO 代码！')
        raise
    except AssertionError as e:
        print(f'测试失败: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e
    except Exception as e:
        print(f'发生错误: {e}')
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_memory_strategy_project()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
def validate_strategy_budget(budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    required_budget_keys = ['memory_cap_mb', 'min_samples_per_s']
    required_quality_keys = ['max_val_loss']
    missing_keys = [key for key in required_budget_keys if key not in budget]
    missing_keys += [key for key in required_quality_keys if key not in quality_floor]
    return {
        'is_valid': len(missing_keys) == 0,
        'missing_keys': missing_keys,
    }


def summarize_memory_strategy_candidates(candidates: List[Dict[str, float]], budget: Dict[str, float], quality_floor: Dict[str, float]) -> Dict[str, object]:
    feasible: List[Dict[str, float]] = []
    quality_failed = 0

    for candidate in candidates:
        memory_ok = candidate['peak_memory_mb'] <= budget['memory_cap_mb']
        speed_ok = candidate['samples_per_s'] >= budget['min_samples_per_s']
        quality_ok = candidate['val_loss'] <= quality_floor['max_val_loss']
        if not quality_ok:
            quality_failed += 1
        if memory_ok and speed_ok and quality_ok:
            feasible.append(candidate)

    feasible.sort(key=lambda x: (x['peak_memory_mb'], -x['samples_per_s'], x['val_loss']))
    best_candidate = feasible[0]['name'] if feasible else None
    return {
        'candidate_count': len(candidates),
        'feasible_count': len(feasible),
        'best_candidate': best_candidate,
        'quality_failed_count': quality_failed,
        'feasible_names': [item['name'] for item in feasible],
    }


def decide_memory_strategy_project(summary: Dict[str, object]) -> Dict[str, object]:
    feasible_count = summary['feasible_count']
    best_candidate = summary['best_candidate']
    quality_failed_count = summary['quality_failed_count']

    if feasible_count == 0:
        return {
            'decision': 'reject',
            'reason': 'no_strategy_meets_budget_and_quality',
            'next_action': 'rework_checkpoint_or_offload_scope',
        }
    if best_candidate in {'checkpoint', 'offload', 'hybrid'}:
        return {
            'decision': 'accept',
            'reason': 'strategy_is_best_feasible_option',
            'next_action': 'promote_to_training_run',
        }
    if quality_failed_count > 0:
        return {
            'decision': 'tune',
            'reason': 'strategy_needs_quality_recovery',
            'next_action': 'adjust_checkpoint_granularity_or_offload_scope',
        }
    return {
        'decision': 'tune',
        'reason': 'baseline_still_best_under_current_budget',
        'next_action': 'revisit_strategy_mix',
    }

```

### 解析

**1. TODO 1: 检查预算与质量阈值**
- **实现方式**：先把显存上限、吞吐下限和验证损失上限检查齐，再进入方案比较。
- **关键点**：没有统一预算口径时，checkpoint / offload / hybrid 之间的比较都没有解释力。
- **项目意义**：这一步把 `76` 固定成预算约束下的显存策略对比页，而不是泛技巧列表。

**2. TODO 2: 汇总显存策略候选**
- **实现方式**：按 peak memory、samples/s 和 val loss 统一过滤候选，再选出最省显存的可行方案。
- **关键点**：显存收益只有在质量和吞吐都没有跌出边界时，才值得被保留。
- **项目意义**：这一步把 `19 / 42 / 43 / 75` 的机制与预算知识收成真正可比较的工程候选。

**3. TODO 3: 输出项目结论**
- **实现方式**：把候选可行性和最优方案统一收成 `accept / tune / reject`。
- **关键点**：项目结论必须回答“当前预算下哪种显存策略值得继续采用”，而不是只输出一个峰值显存最小值。
- **项目意义**：这一步把 `76` 收成显存优化路线中的正式策略对比项目。
