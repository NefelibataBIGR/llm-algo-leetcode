# 85. GRPO Groupwise Alignment Project | GRPO 组内对齐项目

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/85_GRPO_Groupwise_Alignment_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 前置阅读

- [16](./16_GRPO_Loss_Tutorial.md)：先把组内相对优化的公式看懂。
- [2.4](./2_4.md)：回看 GRPO 在对齐技术组页中的位置。
- [50](./50_Preference_Data_and_Evaluation.md)：先把 group-wise 数据与指标口径收好。
### Step 1: 定义项目目标

本项目的重点是验证：同一 prompt 下的多个候选答案，能否通过组内优势比较稳定地拉开排序，并形成可执行的选择结果。
### Step 2: 组织 group-wise batch 与指标

先把同一 prompt 的候选答案按 `group_id` 分组，再计算组内均值、标准差和优势分数。
### Step 3: 项目报告与决策

最后输出组内排名摘要和 `accept / tune / reject`。若组内分布太散，优先检查奖励设计和候选生成，而不是盲目改裁剪范围。
### 提示

- `group_id` 相同的一组样本要放在一起看。
- `mean` 负责看组内整体水平，`std` 负责看组内是否太散。
- 这里的项目页只做最小组内统计，不做真实训练。
- `recommend_grpo_decision` 只需把统计摘要映射到 `accept / tune / reject`。

```python
import torch


def summarize_groupwise_rewards(rewards, group_ids):
    # TODO 1: 将输入转换为 tensor
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    group_ids = torch.as_tensor(group_ids, dtype=torch.long)

    summary = {}
    # TODO 2: 按 group_id 聚合
    for gid in torch.unique(group_ids).tolist():
        mask = group_ids == gid
        group_rewards = rewards[mask]
        # TODO 3: 计算每组 count / mean / std
        summary[int(gid)] = {
            "count": int(mask.sum().item()),
            "mean": float(group_rewards.mean().item()),
            "std": float(group_rewards.std(unbiased=False).item()) if group_rewards.numel() > 1 else 0.0,
        }
    return summary


def recommend_grpo_decision(summary, min_group_count=2):
    # TODO 1: 空 summary 直接 reject
    if not summary:
        return "reject"

    avg_count = sum(v["count"] for v in summary.values()) / len(summary)
    avg_std = sum(v["std"] for v in summary.values()) / len(summary)

    # TODO 2: group 数量与稳定性都够 -> accept
    if avg_count >= min_group_count and avg_std < 1.0:
        return "accept"
    # TODO 3: 有数据但还不稳 -> tune
    if avg_count >= 1:
        return "tune"
    # TODO 4: 其他情况 -> reject
    return "reject"
```

### 测试

运行下面的测试，检查你的组内统计和决策是否正确。

```python
def test_grpo_project():
    rewards = [0.2, 0.5, 0.9, 0.1, 0.3, 0.8]
    group_ids = [0, 0, 0, 1, 1, 1]
    summary = summarize_groupwise_rewards(rewards, group_ids)
    assert summary[0]["count"] == 3
    assert summary[1]["count"] == 3
    assert summary[0]["mean"] > summary[1]["mean"] or summary[1]["mean"] > summary[0]["mean"]
    assert recommend_grpo_decision(summary) == "accept"

    weak_summary = {
        0: {"count": 1, "mean": 0.3, "std": 1.5},
        1: {"count": 1, "mean": 0.2, "std": 1.2},
    }
    assert recommend_grpo_decision(weak_summary) == "tune"

    assert recommend_grpo_decision({}) == "reject"
    print("grpo project passed")


test_grpo_project()
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

GRPO 项目的关键不是把公式写复杂，而是把候选组、奖励和决策口径固定下来。

### 代码

```python
import torch


def summarize_groupwise_rewards(rewards, group_ids):
    # TODO 1: 将输入转换为 tensor
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    group_ids = torch.as_tensor(group_ids, dtype=torch.long)

    summary = {}
    # TODO 2: 按 group_id 聚合
    for gid in torch.unique(group_ids).tolist():
        mask = group_ids == gid
        group_rewards = rewards[mask]
        # TODO 3: 计算每组 count / mean / std
        summary[int(gid)] = {
            "count": int(mask.sum().item()),
            "mean": float(group_rewards.mean().item()),
            "std": float(group_rewards.std(unbiased=False).item()) if group_rewards.numel() > 1 else 0.0,
        }
    return summary


def recommend_grpo_decision(summary, min_group_count=2):
    # TODO 1: 空 summary 直接 reject
    if not summary:
        return "reject"

    avg_count = sum(v["count"] for v in summary.values()) / len(summary)
    avg_std = sum(v["std"] for v in summary.values()) / len(summary)

    # TODO 2: group 数量与稳定性都够 -> accept
    if avg_count >= min_group_count and avg_std < 1.0:
        return "accept"
    # TODO 3: 有数据但还不稳 -> tune
    if avg_count >= 1:
        return "tune"
    # TODO 4: 其他情况 -> reject
    return "reject"
```

### 解析

**1. TODO 1-3：先把 group 统计做实**
- 先把输入转成 tensor，保证后面计算一致。
- 再按 `group_id` 聚合。
- `count / mean / std` 是最小的组内统计三件套。

**2. TODO 1-4：决策看稳定性，而不是只看均值**
- `avg_count` 反映每组样本数量是否足够。
- `avg_std` 反映组内奖励是否过散。
- 组内既有样本量又不太散，才更适合 `accept`。

**3. 这页的定位**
- 它是 GRPO 的最小项目闭环。
- 重点是把组内统计口径和选择决策固定下来。