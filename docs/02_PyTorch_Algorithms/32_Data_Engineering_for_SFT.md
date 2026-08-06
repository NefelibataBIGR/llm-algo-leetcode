# 32. Data Engineering for SFT | SFT 数据工程

**难度：** Placeholder | **环境：** CPU-first | **标签：** `占位`, `SFT`, `data` | **目标人群：** 该小节后续承接的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

占位页，用于后续补齐 SFT 数据工程、清洗和样本组织。

**占位说明：** 32 先占位，后续承接 SFT 数据工程内容。

## Step 1: 定义 SFT 数据结构

先回答一个问题：你的 SFT 数据到底能不能被稳定地转成训练样本？

- 明确字段结构，至少区分 prompt、response、metadata 和可选辅助字段。
- 统一样本边界和标签口径，避免把无效内容误算进监督信号。
- 先统计样本规模、缺失字段、空 response 和重复样本。

## Step 2: 清洗、切分与对齐

数据工程的任务不是只做清理，而是让数据可以稳定进入训练闭环。

- 处理长度超限、格式不一致和重复样本。
- 检查 prompt / response 的切分位置，确保监督只落在预期区域。
- 如果要做 packing 或 template 统一，必须先把规则写成显式口径。

## Step 3: 做最小数据审计

先用最小审计项确认数据可信，再考虑扩规模。

- 记录长度分布、类别分布和异常样本。
- 抽样核对 `input_ids / attention_mask / labels` 是否和预期一致。
- 如果审计结果不稳定，先修数据，不要直接调训练超参。

## Step 4: 输出数据工程结论

最后把数据工程产物变成可以交付的训练输入。

- 输出数据清洗前后对比、样本统计和异常清单。
- 说明哪些规则必须保留，哪些规则只适合当前任务。
- 给出下一步动作：继续清洗、补标注、改模板，还是进入训练闭环。

## STOP HERE

先确认数据能不能被稳定读进训练流程，再谈训练效果。

## 参考代码与解析

```python
def audit_sft_samples(*args, **kwargs):
    raise NotImplementedError

def build_sft_dataset_report(*args, **kwargs):
    raise NotImplementedError
```
