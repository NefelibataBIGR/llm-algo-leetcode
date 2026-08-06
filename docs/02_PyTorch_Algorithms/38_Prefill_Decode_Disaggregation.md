# 38. Prefill Decode Disaggregation | PD 分离

**难度：** Placeholder | **环境：** CPU-first | **标签：** `占位`, `inference`, `pd-disaggregation` | **目标人群：** 该小节后续承接的学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/38_Prefill_Decode_Disaggregation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

占位页，用于后续补齐 prefill / decode 分离的服务架构与调度思路。

**占位说明：** 38 先占位，后续承接 PD 分离内容。

## Step 1: 定义 PD 分离的问题

先回答一个问题：prefill 和 decode 为什么要分开看，分开做的收益是什么？

- 先固定请求类型、上下文长度和生成长度，区分长 prompt 和长 decode 的不同压力。
- 明确 prefill 主要受算力和访存影响，decode 主要受 KV cache 和调度影响。
- 先写清楚 baseline 的服务路径和瓶颈判断。

## Step 2: 组织分离后的服务视角

PD 分离的核心不是拆名字，而是拆资源路径。

- 分别描述 prefill 阶段和 decode 阶段的指标。
- 明确请求在两个阶段之间如何迁移、排队和调度。
- 如果需要共享缓存或共享模型副本，要写出资源边界。

## Step 3: 比较收益与代价

分离之后是否真的更快，要看是否把资源用在了更合适的阶段。

- 观察 TTFT、TPOT、吞吐、显存和调度复杂度。
- 如果 prefill 更快但 decode 更慢，要说明是否只是把压力转移了。
- 如果服务更复杂但收益不明显，就不该强行推进。

## Step 4: 输出部署判断

最后把 PD 分离收成可执行建议。

- 输出 baseline vs PD 分离的对比表。
- 写清楚适合在线、离线还是长上下文服务。
- 给出是否值得进入下一轮架构改造的结论。

## STOP HERE

先把 prefill 和 decode 的责任边界画清楚，再决定要不要拆。

## 参考代码与解析

```python
def summarize_pd_metrics(*args, **kwargs):
    raise NotImplementedError

def compare_pd_strategy(*args, **kwargs):
    raise NotImplementedError
```
