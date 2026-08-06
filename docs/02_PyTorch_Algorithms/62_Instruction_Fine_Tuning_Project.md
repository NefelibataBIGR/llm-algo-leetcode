# 62. Instruction Fine Tuning Project | 指令微调项目

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `SFT`, `Data Engineering` | **目标人群：** 指令数据处理与微调工程

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

指令微调项目的核心不是单纯把样本喂进训练循环，而是确认 instruction、input、response 的结构是否稳定，数据是否重复或失衡，以及训练完成后能不能用统一标准做评估。本节把指令微调收成一个可交付的项目页：先做数据审计，再检查 prompt/response 结构，最后把训练结论整理成可复用的报告。

## 前置阅读

**导语：** 先看 SFT 训练循环、LoRA、学习率调度和端到端报告，再做指令微调项目；这页重点是数据工程和项目收口。
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)

### Step 1: 定义指令微调目标
先回答一个问题：这次微调要提升的是指令遵循、格式稳定性，还是领域知识覆盖？

- 固定底座模型、训练数据、prompt 模板、batch size、seq len 和训练步数。
- 明确 evaluation set 的构成，保证训练集和验证集的分工清晰。
- 记录 instruction、input、response 的字段约定，避免样本格式漂移。
- 先设定数据质量阈值，再决定哪些样本可以进入训练。

#### 图解：09-13 如何收束到 62 指令微调项目

`62` 把 SFT 数据工程和训练闭环组合成一个项目交付模板。

```text
09 SFT data       instruction / input / response / labels
      │
10 LoRA           optional adapter tuning for instruction task
      │
11 Scheduler      lr schedule counted by optimizer update
      │
13 E2E report     train loss / val loss / instruction quality
      │
      ▼
62 Instruction    data audit + format check + project conclusion
```

项目页最小产物：


```python
from typing import Dict, List

```


```python
# TODO: 完成指令数据审计、格式检查和项目总结
# 目标：把 instruction / input / response 数据整理成统一项目报告

def summarize_instruction_dataset(records, max_prompt_chars):
    # ==========================================
    # TODO 1: 审计指令数据集
    # 提示：检查样本数、空 response、重复样本和超长样本。
    # ==========================================
    return {
        'total_samples': 0,
        'empty_response_count': 0,
        'duplicate_count': 0,
        'over_length_count': 0,
        'avg_prompt_chars': 0.0,
    }

def check_instruction_format(batch):
    # ==========================================
    # TODO 2: 核对 prompt / response 结构
    # 提示：检查字段是否齐全、顺序是否稳定、回答是否进入监督范围。
    # ==========================================
    return {
        'valid_count': 0,
        'missing_field_count': 0,
        'format_issue_count': 0,
    }

def build_instruction_project_report(summary, format_check):
    # ==========================================
    # TODO 3: 生成项目结论
    # 提示：把数据质量和格式稳定性合成一段可交付结论。
    # ==========================================
    return {
        'project_ready': False,
        'summary': summary,
        'format_check': format_check,
    }

```


```python
# 测试你的实现
def test_instruction_project_template():
    try:
        records = [
            {'instruction': '解释 LoRA。', 'input': '', 'response': 'LoRA 是低秩适配。'},
            {'instruction': '解释 LoRA。', 'input': '', 'response': 'LoRA 是低秩适配。'},
            {'instruction': '给出答案。', 'input': '', 'response': ''},
        ]
        summary = summarize_instruction_dataset(records, max_prompt_chars=20)
        assert 'total_samples' in summary, '数据审计结果字段缺失！'
        format_check = check_instruction_format(records)
        assert 'valid_count' in format_check, '格式检查结果字段缺失！'
        report = build_instruction_project_report(summary, format_check)
        assert 'project_ready' in report, '项目结论字段缺失！'
        print('测试通过：指令微调项目模板结构正常。')
    except Exception as exc:
        print(f'测试未通过：{exc}')

test_instruction_project_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 审计指令数据集
def summarize_instruction_dataset(records, max_prompt_chars):
    seen = set()
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0
    total_prompt_chars = 0

    for record in records:
        instruction = record.get('instruction', '')
        input_text = record.get('input', '')
        response = record.get('response', '')
        prompt = instruction + input_text
        total_prompt_chars += len(prompt)

        pair = (instruction, input_text, response)
        if not response.strip():
            empty_response_count += 1
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)
        if len(prompt) > max_prompt_chars:
            over_length_count += 1

    total_samples = len(records)
    avg_prompt_chars = total_prompt_chars / total_samples if total_samples else 0.0
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_prompt_chars': avg_prompt_chars,
    }

# TODO 2: 核对 prompt / response 结构
def check_instruction_format(batch):
    valid_count = 0
    missing_field_count = 0
    format_issue_count = 0

    for record in batch:
        if 'instruction' not in record or 'response' not in record:
            missing_field_count += 1
            continue
        if not record.get('instruction', '').strip() or not record.get('response', '').strip():
            format_issue_count += 1
            continue
        valid_count += 1

    return {
        'valid_count': valid_count,
        'missing_field_count': missing_field_count,
        'format_issue_count': format_issue_count,
    }

# TODO 3: 生成项目结论
def build_instruction_project_report(summary, format_check):
    project_ready = summary['empty_response_count'] == 0 and format_check['format_issue_count'] == 0
    return {
        'project_ready': project_ready,
        'summary': summary,
        'format_check': format_check,
    }

```

### 解析

**1. TODO 1: 审计指令数据集**
- **实现方式**：统计样本数、空 response、重复样本和超长样本，并计算 prompt 平均长度。
- **关键点**：指令微调最怕数据格式不稳定。样本重复、空回答和超长 prompt 都会直接影响训练质量。
- **项目意义**：把“能不能训”提前转成“数据是否值得训”。

**2. TODO 2: 核对 prompt / response 结构**
- **实现方式**：检查字段是否齐全，过滤掉缺失 instruction 或 response 的样本，再区分格式问题和完全有效样本。
- **关键点**：训练前就把格式问题筛掉，能减少后续 loss 异常和评估噪声。
- **项目意义**：这一步确保数据工程和训练目标保持一致。

**3. TODO 3: 生成项目结论**
- **实现方式**：把审计结果和格式检查结果合并成一个可交付的 project_ready 判断。
- **关键点**：项目页最终需要的是判断，而不是单纯的统计。
- **项目意义**：这一步把数据检查转成是否进入训练或返工的决策。
