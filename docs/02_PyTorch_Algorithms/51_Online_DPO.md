# 51. Online DPO | 在线 DPO 变体

**难度：** Hard | **环境：** CPU-first | **标签：** `对齐`, `DPO`, `在线学习` | **目标人群：** 后训练与对齐工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/51_Online_DPO.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的章节已经把偏好数据、DPO、GRPO 和离线对齐项目拆开讲过，但真实系统里的问题常常是：偏好数据会持续流入，模型也会持续更新，在线策略是否会把稳定性、成本和收益一起推向可控范围。本节把在线 DPO 变体做成一个概念页：先定义在线问题，再说明在线与离线口径的差异，最后给出最小的推进判断。

这一节不追求工业级训练框架，而是用最小骨架把在线更新、反馈流和评估窗口串起来。

## 前置阅读

**导语：** 先看偏好数据、DPO、GRPO 和对齐评估，再看在线 DPO；这页重点不是重讲损失，而是把在线更新的边界说清楚。
- [44. Preference Data and Evaluation | 偏好数据与评估](./44_Preference_Data_and_Evaluation.md)
- [45. DPO Preference Project | DPO 偏好项目](./45_DPO_Preference_Project.md)
- [46. GRPO Groupwise Alignment Project | GRPO 分组对齐项目](./46_GRPO_Groupwise_Alignment_Project.md)
- [84. DPO Preference Project | DPO 偏好项目](./84_DPO_Preference_Project.md)
- [85. GRPO Groupwise Alignment Project | GRPO 分组对齐项目](./85_GRPO_Groupwise_Alignment_Project.md)

### Step 1: 定义在线 DPO 要解决的问题
先回答一个问题：在线 DPO 这里要优化的是偏好胜率、响应稳定性，还是更新速度？

- 固定初始模型、偏好流、更新频率、batch size 和评估窗口。
- 明确在线样本的来源、过滤规则和是否允许重复反馈。
- 记录比较口径：win rate、loss 波动、更新耗时和回滚条件。
- 先定义评估窗口，再判断在线更新是否值得推进。

#### 图解：44-46-84-85 如何收束到 51 在线 DPO

`51` 把偏好数据、离线对齐和项目级验证收成一个在线更新的概念页。

```text
44 Preference      preference data and evaluation baseline
      │
45 DPO project     offline preference alignment
      │
46 GRPO project    groupwise alignment baseline
      │
84 DPO project     project-level preference tuning
      │
85 GRPO project    project-level groupwise alignment
      │
      ▼
51 Online DPO      update stream + stability window + judgment
```

本节最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成在线 DPO 的问题定义、反馈流整理和推进判断
# 目标：把在线偏好更新收成一个清晰的概念模板

def summarize_online_preferences(feedback_stream):
    # ==========================================
    # TODO 1: 汇总在线反馈
    # 提示：统计样本数、好评率、噪声率和更新时间窗口。
    # ==========================================
    return {
        'sample_count': 0,
        'positive_rate': 0.0,
        'noise_rate': 0.0,
        'window_size': 0,
    }

def compare_online_and_offline(offline_metrics, online_metrics):
    # ==========================================
    # TODO 2: 对比在线与离线口径
    # 提示：比较 win rate、loss 波动、更新时间和稳定性。
    # ==========================================
    return {
        'offline_name': offline_metrics.get('name', 'offline'),
        'online_name': online_metrics.get('name', 'online'),
        'win_rate_delta': 0.0,
        'stability_delta': 0.0,
        'update_cost_delta': 0.0,
    }

def should_try_online_dpo(metrics, min_positive_rate):
    # ==========================================
    # TODO 3: 判断是否值得继续
    # 提示：只有当正反馈足够且噪声受控时才继续。
    # ==========================================
    return {
        'continue_online': False,
        'min_positive_rate': min_positive_rate,
    }

```


```python
# 测试你的实现
def test_online_dpo_template():
    try:
        offline = {'name': 'offline', 'win_rate': 0.52, 'stability': 0.82, 'update_cost': 1.0}
        online = {'name': 'online', 'win_rate': 0.58, 'stability': 0.76, 'update_cost': 1.2}
        stream = [
            {'reward': 1, 'noise': 0.1},
            {'reward': 0, 'noise': 0.2},
            {'reward': 1, 'noise': 0.15},
        ]
        summary = summarize_online_preferences(stream)
        assert 'sample_count' in summary, '在线反馈汇总字段缺失！'
        comp = compare_online_and_offline(offline, online)
        assert 'win_rate_delta' in comp and 'stability_delta' in comp, '在线/离线对比字段不完整！'
        decision = should_try_online_dpo(online, min_positive_rate=0.5)
        assert 'continue_online' in decision, '推进判断字段缺失！'
        print('测试通过：在线 DPO 概念页模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_online_dpo_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 汇总在线反馈
def summarize_online_preferences(feedback_stream):
    sample_count = len(feedback_stream)
    if sample_count == 0:
        return {
            'sample_count': 0,
            'positive_rate': 0.0,
            'noise_rate': 0.0,
            'window_size': 0,
        }

    positive_rate = sum(1 for item in feedback_stream if item.get('reward', 0) > 0) / sample_count
    noise_rate = sum(item.get('noise', 0.0) for item in feedback_stream) / sample_count
    return {
        'sample_count': sample_count,
        'positive_rate': positive_rate,
        'noise_rate': noise_rate,
        'window_size': sample_count,
    }

# TODO 2: 对比在线与离线口径
def compare_online_and_offline(offline_metrics, online_metrics):
    return {
        'offline_name': offline_metrics.get('name', 'offline'),
        'online_name': online_metrics.get('name', 'online'),
        'win_rate_delta': online_metrics.get('win_rate', 0.0) - offline_metrics.get('win_rate', 0.0),
        'stability_delta': online_metrics.get('stability', 0.0) - offline_metrics.get('stability', 0.0),
        'update_cost_delta': online_metrics.get('update_cost', 0.0) - offline_metrics.get('update_cost', 0.0),
    }

# TODO 3: 判断是否值得继续
def should_try_online_dpo(metrics, min_positive_rate):
    return {
        'continue_online': metrics.get('win_rate', 0.0) >= min_positive_rate and metrics.get('stability', 0.0) >= 0.75,
        'min_positive_rate': min_positive_rate,
    }

```

### 解析

**1. TODO 1: 汇总在线反馈**
- **实现方式**：统计样本数、正反馈率、噪声率和窗口大小。
- **关键点**：在线 DPO 的数据不是静态集，而是持续变化的反馈流。
- **项目意义**：这一步把反馈流收成可解释的概念统计。

**2. TODO 2: 对比在线与离线口径**
- **实现方式**：比较在线与离线在胜率、稳定性和更新成本上的差异。
- **关键点**：在线方法的收益不能脱离稳定性和更新代价单独看。
- **项目意义**：帮助判断在线策略是否只是“更快更新”，还是确实更优。

**3. TODO 3: 判断是否值得继续**
- **实现方式**：用正反馈率和稳定性门槛决定是否继续推进。
- **关键点**：在线策略必须有清晰的停止条件，否则会把训练过程变成无界循环。
- **项目意义**：把概念页收束成是否继续试验的最小判断。
