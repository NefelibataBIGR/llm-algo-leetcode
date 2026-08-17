# 38. Prefill Decode Disaggregation | PD 分离
**难度：** Medium | **环境：** CPU-first | **标签：** `推理优化`, `推理服务`, `PD Disaggregation` | **目标人群：** 推理优化学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/38_Prefill_Decode_Disaggregation.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

PD 分离不是为了把系统切得更复杂，而是为了把 `prefill` 和 `decode` 两种完全不同的压力拆开。前者更偏算力和带宽，后者更偏 KV cache 驻留和调度。如果把两者混在一个资源池里，系统常常会既吃不满算力，又挤爆缓存预算。`34`、`36`、`37` 分别把 prefix、decode 和 KV cache 的局部优化讲清了，而 `38` 进一步回答：当这些局部压力已经明显分化时，是否应该把系统级资源池也拆开。

这一节不实现完整 serving 系统，而是先用一个最小教学模拟把三件事固定下来：请求流量如何分层、哪些请求适合进入独立 prefill 池、以及拆池带来的收益是否足以覆盖复杂度上升。学完后，你应该能看清“局部优化 -> 流量分化 -> 系统拆池”这条判断链，而不是把 PD 分离误解成纯架构口号。

**关键词：** `prefill`, `decode`, `queue`, `kv residency`

---

## 前置阅读

- [34. Prefix Caching and Chunked Prefill | 前缀缓存与分块预填充](./34_Prefix_Caching_and_Chunked_Prefill.md)
- [36. Decode Scheduling | Decode 调度](./36_Decode_Scheduling.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)

## 相关阅读

- [39. Inference Fallback and Tiers | 推理分层与回退策略](./39_Inference_Fallback_and_Tiers.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [70. Serving Scheduler Benchmark | 推理服务调度基准](./70_Serving_Scheduler_Benchmark.md)

---

### Step 1: 先分清 prefill 和 decode 的压力来源

- `prefill` 主要看长 prompt 的 attention 计算和访存。
- `decode` 主要看短步迭代、KV cache 驻留和排队。
- 如果一个系统同时被长 prompt 和高并发 decode 拖住，就需要考虑拆池。

### Step 2: 把请求流量写成显式调度账本

![Prefill-Decode Disaggregation](/02_PyTorch_Algorithms/38_pd_disaggregation.svg)

- 记录请求类型、prompt 长度、生成长度和是否命中缓存。
- 分开统计 prefill 队列和 decode 队列的积压。
- 再决定是否值得把两类工作负载拆到不同 worker 上。

### Step 3: 用收益和代价一起判断是否拆分

- 收益至少要体现在更高吞吐、更低排队或更稳的延迟上。
- 代价则包括跨池通信、状态同步和资源碎片化。
- 真正值得保留的，是收益能覆盖系统复杂度上升的方案。

### Step 4: 动手实战

1. 补全 `summarize_request_mix`，统计请求流量构成。
2. 补全 `plan_pd_split`，判断哪些请求适合放入 prefill 池。
3. 补全 `evaluate_pd_decision`，输出拆分结论。

### 提示

- `TODO 1` 建议按这个顺序做：先看 `prompt_tokens` 和 `decode_tokens`，再把请求分成 `prefill_heavy / decode_heavy / mixed`。
- `TODO 2` 不要直接算指标，只要把请求名分配到 `prefill_pool / decode_pool / shared_pool` 三个池里。
- `TODO 3` 先算吞吐变化和延迟变化，再决定 `keep_split` 是否成立。


```python
from typing import Dict, List

```


```python
def summarize_request_mix(requests: List[Dict[str, int]], long_prompt_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 统计请求流量构成。

    你需要返回：
    - prefill_heavy: 长 prompt、短 decode 的请求数
    - decode_heavy: 短 prompt、长 decode 的请求数
    - mixed: 其他请求数
    """
    # 提示：先创建结果字典，再逐个请求判断属于哪一类。
    # result = ???
    # if ???:
    #     result['prefill_heavy'] += 1
    # elif ???:
    #     result['decode_heavy'] += 1
    # else:
    #     result['mixed'] += 1
    raise NotImplementedError


def plan_pd_split(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, object]:
    """
    TODO 2: 把请求分配到 prefill / decode / shared 三个池。
    """
    # 提示：先创建三个列表，再根据 prompt/decode 长度把 request name 放进去。
    # prefill_pool = ???
    # decode_pool = ???
    # shared_pool = ???
    raise NotImplementedError


def evaluate_pd_decision(baseline: Dict[str, float], split_run: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 3: 输出 PD 分离决策。
    """
    # 提示：先算 throughput_gain 和 latency_delta_ms，再决定 keep_split。
    # throughput_gain = ???
    # latency_delta_ms = ???
    # keep_split = ???
    raise NotImplementedError

```


```python
def test_pd_disaggregation_template():
    try:
        requests = [
            {'name': 'a', 'prompt_tokens': 4000, 'decode_tokens': 64},
            {'name': 'b', 'prompt_tokens': 256, 'decode_tokens': 512},
            {'name': 'c', 'prompt_tokens': 1500, 'decode_tokens': 128},
        ]
        summary = summarize_request_mix(requests, long_prompt_threshold=2048)
        assert summary == {'prefill_heavy': 1, 'decode_heavy': 1, 'mixed': 1}
        plan = plan_pd_split(requests, long_prompt_threshold=2048, long_decode_threshold=256)
        assert plan['prefill_pool'] == ['a']
        assert plan['decode_pool'] == ['b']
        assert plan['shared_pool'] == ['c']
        decision = evaluate_pd_decision(
            {'throughput': 100, 'p95_latency_ms': 180},
            {'throughput': 126, 'p95_latency_ms': 150},
        )
        assert decision['throughput_gain'] == 26
        assert decision['latency_delta_ms'] == -30
        assert decision['keep_split'] is True
        print('测试通过：PD 分离模板可以工作。')
    except NotImplementedError:
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError) as e:
        raise NotImplementedError('请先完成 TODO 代码！') from e


test_pd_disaggregation_template()

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
def summarize_request_mix(requests: List[Dict[str, int]], long_prompt_threshold: int) -> Dict[str, int]:
    """
    TODO 1: 统计请求流量构成。

    你需要返回：
    - prefill_heavy: 长 prompt、短 decode 的请求数
    - decode_heavy: 短 prompt、长 decode 的请求数
    - mixed: 其他请求数
    """
    # 提示：先创建结果字典，再逐个请求判断属于哪一类。
    # result = ???
    # if ???:
    #     result['prefill_heavy'] += 1
    # elif ???:
    #     result['decode_heavy'] += 1
    # else:
    #     result['mixed'] += 1
    result = {'prefill_heavy': 0, 'decode_heavy': 0, 'mixed': 0}
    for request in requests:
        prompt_tokens = request.get('prompt_tokens', 0)
        decode_tokens = request.get('decode_tokens', 0)
        if prompt_tokens > long_prompt_threshold and decode_tokens <= long_prompt_threshold // 8:
            result['prefill_heavy'] += 1
        elif decode_tokens > long_prompt_threshold // 8 and prompt_tokens <= long_prompt_threshold:
            result['decode_heavy'] += 1
        else:
            result['mixed'] += 1
    return result


def plan_pd_split(requests: List[Dict[str, int]], long_prompt_threshold: int, long_decode_threshold: int) -> Dict[str, object]:
    """
    TODO 2: 把请求分配到 prefill / decode / shared 三个池。
    """
    # 提示：先创建三个列表，再根据 prompt/decode 长度把 request name 放进去。
    # prefill_pool = ???
    # decode_pool = ???
    # shared_pool = ???
    prefill_pool = []
    decode_pool = []
    shared_pool = []
    for request in requests:
        name = request.get('name', 'request')
        if request.get('prompt_tokens', 0) > long_prompt_threshold and request.get('decode_tokens', 0) <= long_decode_threshold:
            prefill_pool.append(name)
        elif request.get('decode_tokens', 0) > long_decode_threshold and request.get('prompt_tokens', 0) <= long_prompt_threshold:
            decode_pool.append(name)
        else:
            shared_pool.append(name)
    return {'prefill_pool': prefill_pool, 'decode_pool': decode_pool, 'shared_pool': shared_pool}


def evaluate_pd_decision(baseline: Dict[str, float], split_run: Dict[str, float]) -> Dict[str, object]:
    """
    TODO 3: 输出 PD 分离决策。
    """
    # 提示：先算 throughput_gain 和 latency_delta_ms，再决定 keep_split。
    # throughput_gain = ???
    # latency_delta_ms = ???
    # keep_split = ???
    throughput_gain = split_run.get('throughput', 0.0) - baseline.get('throughput', 0.0)
    latency_delta_ms = split_run.get('p95_latency_ms', 0.0) - baseline.get('p95_latency_ms', 0.0)
    return {'throughput_gain': throughput_gain, 'latency_delta_ms': latency_delta_ms, 'keep_split': throughput_gain > 0 and latency_delta_ms <= 0}

```

### 解析

**1. TODO 1：统计请求流量构成**
- 先读取每个请求的 `prompt_tokens` 和 `decode_tokens`，再把请求分成 `prefill_heavy / decode_heavy / mixed`。
- 这一步的意义不是精确建模，而是先判断流量是不是已经明显异构，是否存在拆池动机。

**2. TODO 2：把请求分配到 prefill / decode / shared 三个池**
- `prefill_pool` 放长 prompt、短 decode 的请求；`decode_pool` 放短 prompt、长 decode 的请求；其余进入 `shared_pool`。
- 这里先做最小分配账本，不涉及真实 worker 调度，只固定“谁更适合被单独隔离”。

**3. TODO 3：输出 PD 分离决策**
- 先计算 `throughput_gain` 和 `latency_delta_ms`，再判断拆分后的收益是否同时覆盖吞吐和延迟两侧。
- 这一步回答的是“拆池值不值得保留”，而不是“系统能不能拆”。

**4. 这页的定位**
- PD 分离首先回答“流量是否真的异构”，再回答“拆池是否值得”。
- 请求分类、资源分配和收益评估，构成了最小判断链路。
