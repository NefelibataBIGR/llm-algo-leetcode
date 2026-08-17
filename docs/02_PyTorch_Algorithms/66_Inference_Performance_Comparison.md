# 66. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理优化`, `基准对比`, `性能对比` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一节对应的真实项目问题不是“某个推理优化技巧能不能带来收益”，而是“在既定 workload、延迟约束和显存预算下，哪种推理方案最值得采用”。真实工程里，读者真正要判断的不是单点 latency 或单次吞吐，而是 workload、backend、batch、cache policy 和评测口径固定之后，baseline 与 candidate 是否还能做出可解释的选型结论。

本节的核心矛盾是吞吐、延迟与显存预算之间的权衡：有的方案能压低 TTFT，有的方案能提高 throughput，还有的方案能节省 peak memory，但这些收益未必能同时成立。做完这一节，你应该能输出一份 baseline vs candidate 的推理选型结论，而不只是收集几组 benchmark 数字。

因此，这一页把推理性能对比收成一个最小项目交付入口：先固定 workload 与 baseline，再拆开 prefill 和 decode 记录指标，用统一口径诊断瓶颈、比较候选方案，并把结论收成 `accept / tune / reject` 的项目报告。它直接承接 `20 / 21 / 22` 和 `P1:11` 的推理机制与 KV cache 直觉，并继续通向 `68` 的推测解码基准和 `67` 的量化推理部署。

**关键词：** `benchmark`, `TTFT`, `TPOT`, `throughput`, `KV cache`

---
## 前置阅读

**导语：** 先把解码、KV cache 和推理后端的最小口径理顺，再做推理性能对比；本节不重复讲每个优化机制，而是把它们放到同一个 benchmark 口径里比较。
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)

## 相关阅读

**导语：** 做完基础推理对比后，最自然的下一步是继续拆具体优化收益，或把结论推进到量化部署。
- [68. Speculative Decoding Benchmark | 推测解码基准](./68_Speculative_Decoding_Benchmark.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)

---
### Step 1: 定义推理对比项目目标
先回答一个问题：在同一模型、同一输入集和同一硬件环境下，哪种推理策略更划算？

- 固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和评测轮数。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length + warm-up + 多轮测量` 开始。
- 明确 candidate 只改一个变量，例如 FlashAttention、batch size、KV cache 策略、量化精度或推理后端。
- 统一核心指标：TTFT、TPOT、generated tokens/s、total latency、peak memory。
- 这节的目标不是证明某个方案“能跑”，而是在相同约束下输出可解释的推理选型结论。

### Step 2: 先确认 workload 和 baseline 口径合法

推理对比必须先确认 workload 和 baseline 可复现，不能直接把不同 prompt、不同 batch 或不同 backend 的数字放在一起比较。

- 先固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和 warm-up / 多轮测量方式。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length` 开始，保证后续 candidate 的改动边界清晰。
- TTFT、TPOT、throughput、total latency 和 peak memory 必须来自同一套 workload，避免把不同实验口径拼成一张表。
- 如果 baseline 自身波动很大，后面的 candidate 结果就没有解释空间。

### Step 3: 用统一口径比较收益与成本

推理项目必须用统一口径同时看 latency、throughput 和 memory，不能只挑单一指标下结论。

| 瓶颈类型 | 典型信号 | 候选方向 |
| --- | --- | --- |
| prefill-bound | prefill 占比高，长 prompt 变慢明显 | FlashAttention、chunked prefill、batching |
| decode-bound | TPOT 高，decode 占比高 | speculative decoding、multi-token decoding、decode scheduling |
| memory-bound | peak memory 接近预算，batch 上不去 | KV cache quantization、PagedAttention、GQA/MQA |
| balanced | 各项都不突出 | 保持 baseline 或做小步 profiling |

### Step 4: 输出推理选型结论

推理选型最终不是输出“哪个 benchmark 更好看”，而是输出哪种方案值得在当前 workload 下继续保留、微调或切换。

- 输出 baseline vs candidate 对比表，至少包含 TTFT、TPOT、throughput、total latency、peak memory 和瓶颈判断。
- 如果 candidate 只提升吞吐但明显拉高 TTFT，要说明适合离线批处理还是在线交互。
- 如果 candidate 显存更省但 TPOT 变差，要说明是否为了更大 batch 或更长上下文让路。
- 最终决策统一使用 `accept / tune / reject`：候选方案值得采用、还需继续调优、或当前不值得切换。
- 报告结论必须回扣 Step 1 的 workload，不能泛化成“某方案永远更好”。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整推理性能对比项目流程。下面的代码实现六块最小能力：配置 workload、汇总 prefill/decode、计算指标、诊断瓶颈、比较候选方案和输出决策。
#### 图解：推理项目如何从 workload 走到选型结论

`66` 不重复讲所有推理优化机制，而是把它们放进同一套 benchmark 口径里比较。

```text
workload config
      │
      ▼
baseline run ──► prefill/decode metrics ──► bottleneck diagnosis
      │                                               │
      ▼                                               ▼
candidate run ─► candidate comparison ───────────► accept / tune / reject
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| Workload | backend、batch、prompt tokens、generated tokens、dtype、cache policy | 保证可复现 |
| 指标 | TTFT、TPOT、throughput、total latency、peak memory | 保证同口径比较 |
| 诊断 | prefill-bound、decode-bound、memory-bound、balanced | 解释为什么优化有效或无效 |
| 对比 | latency / throughput / memory delta | 判断 candidate 是否值得保留 |
| 决策 | accept / tune / reject | 输出推理选型结论 |


```python
import time

```


```python
# 补全推理性能对比的六个关键函数
# 目标：完成 workload -> metrics -> bottleneck -> comparison -> decision 的最小项目链路

def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    """汇总推理 workload 配置，形成统一比较口径。"""
    # ==========================================
    # TODO 1: 汇总推理 workload 配置
    # 提示：total_tokens = prompt_tokens + generated_tokens
    # ==========================================
    # total_tokens = ???
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    """汇总 prefill / decode 延迟，形成最小延迟摘要。"""
    # ==========================================
    # TODO 2: 汇总 prefill / decode 延迟
    # 提示：TTFT 近似等于 prefill_ms；TPOT = decode_ms / generated_tokens。
    # ==========================================
    # total_ms = ???
    # ttft_ms = ???
    # tpot_ms = ???
    # prefill_share = ???
    # decode_share = ???
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    """把 workload 和延迟摘要收束成统一推理指标。"""
    # ==========================================
    # TODO 3: 计算推理项目核心指标
    # 提示：throughput 表示整个 batch 每秒生成 token 数。
    # ==========================================
    # output_tokens = ???
    # throughput_tok_s = ???
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    """根据显存预算与 prefill/decode 占比诊断推理瓶颈。"""
    # ==========================================
    # TODO 4: 诊断推理瓶颈
    # 规则：显存接近预算优先判 memory-bound；否则按 prefill/decode 占比判断。
    # ==========================================
    # memory_pressure = ???
    # prefill_heavy = ???
    # decode_heavy = ???
    # if ???:
    #     bottleneck = ???
    #     reason = ???
    # elif ???:
    #     bottleneck = ???
    #     reason = ???
    # elif ???:
    #     bottleneck = ???
    #     reason = ???
    # else:
    #     bottleneck = ???
    #     reason = ???
    return {'bottleneck': bottleneck, 'reason': reason}

def compare_inference_candidates(baseline_metrics, candidate_metrics):
    """统一比较 baseline 与 candidate 的推理收益和代价。"""
    # ==========================================
    # TODO 5: 比较 baseline 和 candidate
    # 提示：latency / TTFT / TPOT / memory 的 delta 用 baseline - candidate；throughput gain 用比例增益。
    # ==========================================
    # total_latency_delta_ms = ???
    # ttft_delta_ms = ???
    # tpot_delta_ms = ???
    # peak_mem_delta_mb = ???
    # throughput_gain = ???
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    """根据吞吐、TTFT 和瓶颈类型输出推理选型建议。"""
    # ==========================================
    # TODO 6: 输出推理选型建议
    # 规则：吞吐明显提升且 TTFT 没明显退化则 accept；有收益但仍有瓶颈则 tune；否则 reject。
    # ==========================================
    # throughput_good = ???
    # ttft_ok = ???
    # still_tunable = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    return {'decision': decision, 'reason': reason}

```


```python
# 测试你的实现
def test_inference_project_template():
    try:
        config = build_inference_config(
            model_name='tiny-llama',
            backend='pytorch-eager',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='static-kv-cache',
        )
        assert config['total_tokens'] == 160, "total_tokens 计算不正确！"
        assert config['batch_size'] == 2, "batch_size 应保留原始配置！"

        latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
        assert latency['total_ms'] == 240.0, "total_ms 计算不正确！"
        assert latency['ttft_ms'] == 80.0, "ttft_ms 计算不正确！"
        assert latency['tpot_ms'] == 5.0, "tpot_ms 计算不正确！"
        assert latency['prefill_share'] == 0.333, "prefill_share 计算不正确！"
        assert latency['decode_share'] == 0.667, "decode_share 计算不正确！"

        metrics = compute_inference_metrics(config, latency, peak_mem_mb=4096.0)
        assert metrics['throughput_tok_s'] == 266.67, "throughput_tok_s 计算不正确！"
        assert metrics['peak_mem_mb'] == 4096.0, "peak_mem_mb 记录不正确！"

        memory_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=4400.0)
        assert memory_bound['bottleneck'] == 'memory-bound', "显存接近预算时应优先判为 memory-bound！"

        decode_bound = diagnose_inference_bottleneck(metrics, memory_budget_mb=8192.0)
        assert decode_bound['bottleneck'] == 'decode-bound', "decode 占比高时应判为 decode-bound！"

        candidate_config = build_inference_config(
            model_name='tiny-llama',
            backend='paged-attention',
            batch_size=2,
            prompt_tokens=128,
            generated_tokens=32,
            dtype='fp16',
            cache_policy='paged-kv-cache',
        )
        candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
        candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
        comparison = compare_inference_candidates(metrics, candidate_metrics)

        assert comparison['total_latency_delta_ms'] == 35.0, "total latency delta 计算不正确！"
        assert comparison['ttft_delta_ms'] == -5.0, "TTFT delta 计算不正确！"
        assert comparison['tpot_delta_ms'] == 1.25, "TPOT delta 计算不正确！"
        assert comparison['peak_mem_delta_mb'] == 512.0, "peak memory delta 计算不正确！"
        assert comparison['throughput_gain'] > 0.15, "throughput gain 应体现候选方案收益！"

        decision = recommend_inference_decision(comparison, decode_bound)
        assert decision['decision'] == 'accept', "吞吐提升且 TTFT 未明显退化时应建议 accept！"

        weak_comparison = dict(comparison)
        weak_comparison['throughput_gain'] = 0.02
        weak_comparison['ttft_delta_ms'] = 1.0
        assert recommend_inference_decision(weak_comparison, decode_bound)['decision'] == 'tune', "小幅收益但仍有瓶颈时应建议 tune！"

        bad_comparison = dict(comparison)
        bad_comparison['throughput_gain'] = -0.05
        bad_comparison['ttft_delta_ms'] = -30.0
        assert recommend_inference_decision(bad_comparison, {'bottleneck': 'balanced'})['decision'] == 'reject', "没有收益且 TTFT 退化时应建议 reject！"

        print("✅ 推理性能对比项目模板代码通过基础校验。")

    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError, AssertionError, RuntimeError) as e:
        if isinstance(e, AttributeError):
            print("代码未完成，无法找到必要的属性")
        elif isinstance(e, NameError):
            print("代码可能未完成，导致了变量未定义")
        elif isinstance(e, TypeError):
            print("代码可能未完成，导致了操作错误")
        elif isinstance(e, ValueError):
            print("代码可能未完成，导致了数值错误")
        elif isinstance(e, AssertionError):
            print(f"❌ 测试失败: {e}")
        elif isinstance(e, RuntimeError):
            print("代码可能未完成，导致了运行时错误")
        else:
            print("代码可能未完成，导致了断言失败")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except Exception as e:
        print(f"❌ 发生未知异常: {e}")
        raise


test_inference_project_template()

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
# TODO 1: 汇总推理 workload 配置
def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
    total_tokens = prompt_tokens + generated_tokens
    return {
        'model_name': model_name,
        'backend': backend,
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'total_tokens': total_tokens,
        'dtype': dtype,
        'cache_policy': cache_policy,
    }

# TODO 2: 汇总 prefill / decode 延迟
def summarize_prefill_decode(prefill_ms, decode_ms, generated_tokens):
    total_ms = prefill_ms + decode_ms
    ttft_ms = prefill_ms
    tpot_ms = decode_ms / generated_tokens if generated_tokens else 0.0
    prefill_share = prefill_ms / total_ms if total_ms else 0.0
    decode_share = decode_ms / total_ms if total_ms else 0.0
    return {
        'prefill_ms': round(prefill_ms, 2),
        'decode_ms': round(decode_ms, 2),
        'total_ms': round(total_ms, 2),
        'ttft_ms': round(ttft_ms, 2),
        'tpot_ms': round(tpot_ms, 4),
        'prefill_share': round(prefill_share, 3),
        'decode_share': round(decode_share, 3),
    }

# TODO 3: 计算推理项目核心指标
def compute_inference_metrics(config, latency_summary, peak_mem_mb):
    output_tokens = config['batch_size'] * config['generated_tokens']
    total_seconds = latency_summary['total_ms'] / 1000.0
    throughput_tok_s = output_tokens / total_seconds if total_seconds else 0.0
    return {
        'backend': config['backend'],
        'batch_size': config['batch_size'],
        'prompt_tokens': config['prompt_tokens'],
        'generated_tokens': config['generated_tokens'],
        'ttft_ms': latency_summary['ttft_ms'],
        'tpot_ms': latency_summary['tpot_ms'],
        'throughput_tok_s': round(throughput_tok_s, 2),
        'total_ms': latency_summary['total_ms'],
        'prefill_share': latency_summary['prefill_share'],
        'decode_share': latency_summary['decode_share'],
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 4: 诊断推理瓶颈
def diagnose_inference_bottleneck(metrics, memory_budget_mb=None):
    if memory_budget_mb is not None and metrics['peak_mem_mb'] >= 0.9 * memory_budget_mb:
        bottleneck = 'memory-bound'
        reason = 'peak memory 接近预算，优先检查 KV cache、batch size、量化和分页策略。'
    elif metrics['prefill_share'] >= 0.6:
        bottleneck = 'prefill-bound'
        reason = 'prefill 占比高，优先检查 prompt length、FlashAttention、chunked prefill 和 batching。'
    elif metrics['decode_share'] >= 0.6:
        bottleneck = 'decode-bound'
        reason = 'decode 占比高，优先检查 KV cache 读写、decode scheduling、speculative decoding 或 multi-token decoding。'
    else:
        bottleneck = 'balanced'
        reason = 'prefill、decode 和显存压力都不突出，先保持 baseline 或继续做细粒度 profiling。'
    return {'bottleneck': bottleneck, 'reason': reason}

# TODO 5: 比较 baseline 和 candidate
def compare_inference_candidates(baseline_metrics, candidate_metrics):
    total_latency_delta_ms = baseline_metrics['total_ms'] - candidate_metrics['total_ms']
    ttft_delta_ms = baseline_metrics['ttft_ms'] - candidate_metrics['ttft_ms']
    tpot_delta_ms = baseline_metrics['tpot_ms'] - candidate_metrics['tpot_ms']
    peak_mem_delta_mb = baseline_metrics['peak_mem_mb'] - candidate_metrics['peak_mem_mb']
    throughput_gain = (
        candidate_metrics['throughput_tok_s'] / baseline_metrics['throughput_tok_s'] - 1.0
        if baseline_metrics['throughput_tok_s'] else 0.0
    )
    return {
        'total_latency_delta_ms': round(total_latency_delta_ms, 2),
        'ttft_delta_ms': round(ttft_delta_ms, 2),
        'tpot_delta_ms': round(tpot_delta_ms, 4),
        'peak_mem_delta_mb': round(peak_mem_delta_mb, 2),
        'throughput_gain': round(throughput_gain, 4),
    }

# TODO 6: 输出推理选型建议
def recommend_inference_decision(comparison, candidate_bottleneck, min_throughput_gain=0.1, max_ttft_regression_ms=20.0):
    ttft_regression_ms = -comparison['ttft_delta_ms']
    if comparison['throughput_gain'] >= min_throughput_gain and ttft_regression_ms <= max_ttft_regression_ms:
        decision = 'accept'
        reason = 'candidate 吞吐提升明显，TTFT 退化在可接受范围内，值得进入正式推理方案。'
    elif comparison['throughput_gain'] > 0.0 and candidate_bottleneck['bottleneck'] != 'balanced':
        decision = 'tune'
        reason = 'candidate 已有收益，但瓶颈仍然存在，继续围绕诊断结果调参或换策略。'
    else:
        decision = 'reject'
        reason = 'candidate 收益不足或交互延迟退化明显，当前不值得切换。'
    return {'decision': decision, 'reason': reason}

baseline_config = build_inference_config('tiny-llama', 'pytorch-eager', 2, 128, 32, 'fp16', 'static-kv-cache')
baseline_latency = summarize_prefill_decode(prefill_ms=80.0, decode_ms=160.0, generated_tokens=32)
baseline_metrics = compute_inference_metrics(baseline_config, baseline_latency, peak_mem_mb=4096.0)
print(baseline_config)
print(baseline_metrics)
print(diagnose_inference_bottleneck(baseline_metrics, memory_budget_mb=8192.0))

candidate_config = build_inference_config('tiny-llama', 'paged-attention', 2, 128, 32, 'fp16', 'paged-kv-cache')
candidate_latency = summarize_prefill_decode(prefill_ms=85.0, decode_ms=120.0, generated_tokens=32)
candidate_metrics = compute_inference_metrics(candidate_config, candidate_latency, peak_mem_mb=3584.0)
comparison = compare_inference_candidates(baseline_metrics, candidate_metrics)
print(candidate_metrics)
print(comparison)
print(recommend_inference_decision(comparison, diagnose_inference_bottleneck(candidate_metrics, memory_budget_mb=8192.0)))

```

### 解析

**1. TODO 1: 汇总推理 workload 配置**
- **实现方式**：把模型、backend、batch size、prompt tokens、generated tokens、dtype 和 cache policy 放进同一个配置对象。
- **关键点**：推理 benchmark 的第一原则是固定 workload。没有 workload，TTFT、TPOT、吞吐和显存都没有可比性。
- **项目意义**：后续 baseline 和 candidate 只能改一个变量，否则很难判断收益来自哪里。

**2. TODO 2: 汇总 prefill / decode 延迟**
- **实现方式**：`total_ms = prefill_ms + decode_ms`，TTFT 近似取 `prefill_ms`，TPOT 取 `decode_ms / generated_tokens`。
- **关键点**：prefill 和 decode 的瓶颈不同。总耗时下降不代表交互体验一定变好，TTFT 和 TPOT 必须拆开看。
- **项目意义**：这一步把推理性能从一个笼统 latency 拆成可诊断的两段。

**3. TODO 3: 计算推理项目核心指标**
- **实现方式**：用 `batch_size * generated_tokens / total_seconds` 计算 generated tokens/s，并和 TTFT、TPOT、total latency、peak memory 放在同一张账本里。
- **关键点**：throughput 统计的是整个 batch 的输出 token 产出，不是单条请求的 token 数。
- **项目意义**：同一个 candidate 可能吞吐更高但 TTFT 更差，指标必须一起看。

**4. TODO 4: 诊断推理瓶颈**
- **实现方式**：显存接近预算时优先判为 `memory-bound`；否则用 prefill/decode 占比判断主要瓶颈。
- **关键点**：显存预算是硬约束。如果显存已经接近上限，即使 decode 占比高，也要先处理 KV cache、batch size 或量化。
- **项目意义**：诊断结果决定下一步选 FlashAttention、chunked prefill、PagedAttention、KV cache 量化还是 decode scheduling。

**5. TODO 5: 比较 baseline 和 candidate**
- **实现方式**：latency、TTFT、TPOT 和 peak memory 使用 `baseline - candidate`，正数表示 candidate 更好；throughput 使用比例增益。
- **关键点**：delta 的方向要固定，否则报告容易把退化误写成收益。
- **项目意义**：项目报告不只写绝对值，更要说明 candidate 相比 baseline 改善或退化了多少。

**6. TODO 6: 输出推理选型建议**
- **accept**：吞吐提升达标，TTFT 退化在可接受范围内，说明候选方案值得采用。
- **tune**：candidate 有收益，但瓶颈仍然存在，需要继续沿诊断方向调参。
- **reject**：candidate 收益不足，或交互延迟退化明显，当前不值得切换。
- **项目意义**：推理选型不能只看一个指标。最终结论要同时考虑 workload、吞吐、TTFT、TPOT、显存和瓶颈类型。

**推理性能对比的实验原则**
- **变量控制**：同一轮对比中只改一个变量，例如 batch size、precision、推理后端或 cache 策略。
- **指标闭环**：每次实验至少记录 TTFT、TPOT、throughput 和 peak memory。
- **阶段拆分**：把 prefill 和 decode 分开看，避免把长 prompt 问题误判成 decode 问题。
- **结果复盘**：最终输出要回扣 Step 1 的问题：在给定约束下，哪种推理策略最划算，理由是什么。
