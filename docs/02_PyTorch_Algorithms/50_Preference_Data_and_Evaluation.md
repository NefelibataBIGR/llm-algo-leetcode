# 50. Preference Data and Evaluation | 偏好数据与对齐评测

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/50_Preference_Data_and_Evaluation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*

## 前置阅读

**导语：** 这一节先理解偏好数据和评测口径，再看它如何接到对齐方法。
- [15](./15_DPO_Loss_Tutorial.md)：先理解 chosen / rejected 的偏好对。
- [16](./16_GRPO_Loss_Tutorial.md)：再看 group-wise 的奖励比较。
- [2.4](./2_4.md)：回看对齐技术组页的阅读顺序。
## 相关阅读

- [45](./84_DPO_Preference_Project.md)：把偏好对和 DPO 项目闭环接起来。
- [46](./85_GRPO_Groupwise_Alignment_Project.md)：把 group-wise 数据和 GRPO 项目闭环接起来。
- [2.9](./2_9.md)：回看项目页如何把对齐能力收束成工程验证。
### Step 1: 偏好数据长什么样

典型偏好样本至少包含 `prompt`、`chosen`、`rejected`，有些任务还会带 `source`、`judge`、`score` 和 `group_id`。
### Step 2: 对齐评测看什么

- `win-rate`：新模型胜出的比例。
- `judge score`：打分器或人工判分。
- `pairwise accuracy`：偏好对上选对的比例。
- `safety / style`：额外的对齐约束。
### Step 3: 从样本到指标

偏好对齐最怕两件事：数据本身不干净，以及评测口径不稳定。这里先把数据检查和最小指标汇总函数做出来。
### Step 4: 动手实战

实现一个最小偏好数据汇总器，检查空样本、重复样本和 win-rate 口径。
### 提示

- `empty` 统计的是 `prompt / chosen / rejected` 中任意一项为空的样本数。
- `duplicate_prompt_groups` 统计的是 prompt 重复出现的组数，不是重复样本总数。
- `avg_pair_tokens` 用 `split()` 近似即可，重点是口径一致。
- `compute_win_rate` 只看 `chosen_score > rejected_score`，长度不一致要直接报错。

```python
from collections import Counter
from dataclasses import dataclass

@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def summarize_preference_dataset(samples):
    """
    TODO: 汇总偏好数据的最小统计信息。

    你需要返回：
    - total: 样本总数
    - empty: 空 prompt / chosen / rejected 的样本数
    - duplicate_prompt_groups: prompt 重复的组数
    - avg_pair_tokens: chosen + rejected 的平均 token 近似数（用 split 近似即可）
    """
    # TODO 1: 统计总样本数
    # TODO 2: 检查空样本
    # TODO 3: 统计重复 prompt 组数
    # TODO 4: 统计平均 pair token 数
    raise NotImplementedError


def compute_win_rate(chosen_scores, rejected_scores):
    """
    TODO: 计算最小 win-rate。

    规则：chosen_score > rejected_score 记为一次胜出。
    """
    # TODO 1: 校验长度
    # TODO 2: 统计胜出次数
    # TODO 3: 返回比例
    raise NotImplementedError
```

### 测试

运行下面的测试，检查你的偏好数据汇总和 win-rate 计算是否正确。

```python
def test_preference_summary():
    samples = [
        PreferenceSample("Explain LoRA", "LoRA adds adapters.", "LoRA changes all weights."),
        PreferenceSample("Explain DPO", "DPO uses preference pairs.", "DPO needs no reward model at inference."),
        PreferenceSample("Explain DPO", "Short answer.", ""),
        PreferenceSample("", "missing prompt", "missing rejected"),
    ]
    stats = summarize_preference_dataset(samples)
    assert stats["total"] == 4
    assert stats["empty"] == 2
    assert stats["duplicate_prompt_groups"] == 1
    assert stats["avg_pair_tokens"] > 0
    print("preference summary passed")


def test_win_rate():
    win_rate = compute_win_rate([0.8, 0.7, 0.9], [0.2, 0.5, 0.95])
    assert 0.6 < win_rate < 0.8
    assert compute_win_rate([1.0, 0.9], [0.2, 0.1]) == 1.0
    assert compute_win_rate([0.1, 0.2], [0.3, 0.4]) == 0.0
    try:
        compute_win_rate([0.5], [0.2, 0.1])
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch should raise ValueError")
    print("win rate passed")


test_preference_summary()
test_win_rate()
```

🛑 **STOP HERE** 🛑
## 参考代码与解析

### 代码

```python
from collections import Counter
from dataclasses import dataclass

@dataclass
class PreferenceSample:
    prompt: str
    chosen: str
    rejected: str


def summarize_preference_dataset(samples):
    # TODO 1: 统计总样本数
    total = len(samples)

    # TODO 2: 检查空样本
    empty = sum(
        1
        for s in samples
        if not s.prompt.strip() or not s.chosen.strip() or not s.rejected.strip()
    )

    # TODO 3: 统计重复 prompt 组数
    duplicate_prompt_groups = sum(
        v > 1 for v in Counter(s.prompt for s in samples).values()
    )

    # TODO 4: 统计平均 pair token 数
    avg_pair_tokens = (
        sum(len(s.chosen.split()) + len(s.rejected.split()) for s in samples) / total
        if total else 0.0
    )

    return {
        "total": total,
        "empty": empty,
        "duplicate_prompt_groups": duplicate_prompt_groups,
        "avg_pair_tokens": round(avg_pair_tokens, 2),
    }


def compute_win_rate(chosen_scores, rejected_scores):
    # TODO 1: 校验长度
    if len(chosen_scores) != len(rejected_scores):
        raise ValueError("score length mismatch")

    # TODO 2: 统计胜出次数
    wins = sum(c > r for c, r in zip(chosen_scores, rejected_scores))

    # TODO 3: 返回比例
    return wins / len(chosen_scores) if chosen_scores else 0.0
```

### 解析

**1. TODO 1-4：偏好数据先看结构，再看质量**
- `total` 统计样本总数，先确认数据规模。
- `empty` 统计空 prompt / chosen / rejected，先排除脏样本。
- `duplicate_prompt_groups` 统计重复 prompt 组数，用来发现一题多答的样本聚集。
- `avg_pair_tokens` 用 token 近似数检查样本长度，避免极端长短样本拉偏训练。

**2. TODO 1-3：win-rate 是最轻的对齐评测口径**
- 先校验长度一致，避免比较样本错位。
- 再统计 `chosen_score > rejected_score` 的胜出次数。
- 最后返回胜出比例，作为最小的对齐效果检查。

**3. 这页的定位**
- 这里先固定偏好数据和评测的最小口径。
- 如果后面要接 DPO、GRPO 或更复杂的 judge 评测，再往上扩展即可。