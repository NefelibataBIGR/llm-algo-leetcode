# 84. DPO Preference Project | DPO 偏好优化项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/84_DPO_Preference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 前置阅读

- [15](./15_DPO_Loss_Tutorial.md)：先把 DPO 损失公式和数据形态看懂。
- [2.4](./2_4.md)：回看 DPO 在对齐技术组页中的位置。
- [50](./50_Preference_Data_and_Evaluation.md)：先把偏好数据与评测口径收好。
### Step 1: 定义项目目标

本项目不追求完整训练管线，而是用最小 DPO 闭环验证：偏好对是否能稳定拉开 chosen / rejected 的分数差，并给出是否值得继续投入训练的判断。
### Step 2: 组织偏好 batch 与损失口径

先把偏好对组织成 batch，再计算 DPO loss、平均 margin 和最小评测摘要。
### Step 3: 项目报告与决策

最后用一致的口径输出 `accept / tune / reject`。如果偏好对数据太少或 margin 波动过大，优先回到数据层，而不是继续调学习率。
### 提示

- `compute_dpo_loss` 只需要最小 DPO 公式：policy / ref 的 log-ratio 差值。
- `summarize_dpo_metrics` 重点看 chosen 与 rejected 的 margin 分布，不要把 ref 混进来。
- `recommend_alignment_decision` 只做最小决策：margin 足够大就 `accept`，中间区间 `tune`，否则 `reject`。
- 这里的项目页不做完整训练，只做最小闭环验证。

```python
import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO 1: 计算 policy_logratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    # TODO 2: 计算 ref_logratios
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # TODO 3: 计算 logits
    logits = beta * (policy_logratios - ref_logratios)

    # TODO 4: 返回最小 DPO loss
    return -F.logsigmoid(logits).mean()


def summarize_dpo_metrics(policy_chosen_logps, policy_rejected_logps):
    # TODO 1: 计算 margins
    margins = policy_chosen_logps - policy_rejected_logps

    # TODO 2: 汇总 mean / min / max
    return {
        "margin_mean": float(margins.mean().item()),
        "margin_min": float(margins.min().item()),
        "margin_max": float(margins.max().item()),
    }


def recommend_alignment_decision(metrics, min_margin=0.15):
    # TODO 1: margin_mean 足够大且 margin_min 不太差 -> accept
    if metrics["margin_mean"] >= min_margin and metrics["margin_min"] > -0.2:
        return "accept"
    # TODO 2: margin_mean 非负但不够稳 -> tune
    if metrics["margin_mean"] >= 0.0:
        return "tune"
    # TODO 3: 其他情况 -> reject
    return "reject"
```

### 测试

运行下面的测试，检查你的 DPO 项目闭环是否正确。

```python
def test_dpo_project():
    policy_chosen = torch.tensor([-1.1, -0.9, -1.0])
    policy_rejected = torch.tensor([-1.5, -1.3, -1.2])
    ref_chosen = torch.tensor([-1.4, -1.1, -1.3])
    ref_rejected = torch.tensor([-1.6, -1.4, -1.5])
    loss = compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    metrics = summarize_dpo_metrics(policy_chosen, policy_rejected)
    assert loss.item() > 0
    assert metrics["margin_mean"] > 0
    assert metrics["margin_min"] > 0
    assert recommend_alignment_decision(metrics) == "accept"

    weak_metrics = {"margin_mean": 0.01, "margin_min": -0.05, "margin_max": 0.12}
    assert recommend_alignment_decision(weak_metrics) == "tune"

    bad_metrics = {"margin_mean": -0.2, "margin_min": -0.8, "margin_max": 0.02}
    assert recommend_alignment_decision(bad_metrics) == "reject"
    print("dpo project passed")


test_dpo_project()
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

这个项目页只求把 DPO 的最小工程闭环跑通：数据对、loss 对、摘要对、决策对。

### 代码

```python
import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # TODO 1: 计算 policy_logratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps

    # TODO 2: 计算 ref_logratios
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    # TODO 3: 计算 logits
    logits = beta * (policy_logratios - ref_logratios)

    # TODO 4: 返回最小 DPO loss
    return -F.logsigmoid(logits).mean()


def summarize_dpo_metrics(policy_chosen_logps, policy_rejected_logps):
    # TODO 1: 计算 margins
    margins = policy_chosen_logps - policy_rejected_logps

    # TODO 2: 汇总 mean / min / max
    return {
        "margin_mean": float(margins.mean().item()),
        "margin_min": float(margins.min().item()),
        "margin_max": float(margins.max().item()),
    }


def recommend_alignment_decision(metrics, min_margin=0.15):
    # TODO 1: margin_mean 足够大且 margin_min 不太差 -> accept
    if metrics["margin_mean"] >= min_margin and metrics["margin_min"] > -0.2:
        return "accept"
    # TODO 2: margin_mean 非负但不够稳 -> tune
    if metrics["margin_mean"] >= 0.0:
        return "tune"
    # TODO 3: 其他情况 -> reject
    return "reject"
```

### 解析

**1. TODO 1-4：DPO loss 的核心是 log-ratio 差值**
- 先算 policy / ref 的 chosen-rejected 差值。
- 再做 beta 缩放。
- 最后用 `-logsigmoid` 把它转成最小化目标。

**2. TODO 1-3：项目摘要看 margin 分布**
- `margin_mean` 看整体是否拉开。
- `margin_min` 看最差样本是否掉得太厉害。
- `margin_max` 只是辅助观察。

**3. 这页的定位**
- 它是 DPO 的最小项目闭环，不是完整训练。
- 只要数据、loss、摘要、决策四件事对齐，就算达标。