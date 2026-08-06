# 66. Inference Performance Comparison | 推理性能对比实验

**难度：** Hard | **环境：** CPU-first | **标签：** `推理`, `benchmark`, `profiling` | **目标人群：** 推理工程与性能分析

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

推理优化里最容易出现的问题是只看单点收益：某个方案 latency 更低，另一个方案 throughput 更高，还有一个方案显存更省。如果模型、输入、batch size、精度、KV cache 策略和评测口径没有固定，这些数字很难放在一起比较，也很难支撑工程选型。

本节把推理优化做成一个对比项目：围绕同一个 workload，拆开 prefill 和 decode，记录 TTFT、TPOT、throughput、peak memory 和 cache 策略，再回答“在给定约束下哪种推理方案最划算”。代码区实现最小可复用的项目模板：推理配置、prefill/decode 指标汇总、瓶颈诊断、候选方案比较和最终决策。

**关键词：** `benchmark`, `TTFT`, `TPOT`, `throughput`, `KV cache`

---
## 前置阅读

**导语：** 先理解 Attention、KV cache 和显存层级，再做推理性能对比；本节不重复讲每个优化机制，而是把它们放到同一个 benchmark 口径里比较。
- [04. Attention MHA/GQA | 多头注意力](./04_Attention_MHA_GQA.md)
- [20. FlashAttention Sim | FlashAttention 模拟](./20_FlashAttention_Sim.md)
- [21. Decoding Strategies | 解码策略](./21_Decoding_Strategies.md)
- [22. vLLM PagedAttention | vLLM 分页注意力](./22_vLLM_PagedAttention.md)
- [P1: 03. GPU Architecture and Memory | GPU 物理架构与内存层级](../01_Hardware_Math_and_Systems/03_GPU_Architecture_and_Memory.md)
- [P1: 11. KV Cache and Memory Growth | KV Cache 与显存增长](../01_Hardware_Math_and_Systems/11_KV_Cache_and_Memory_Growth.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读

**导语：** 如果要继续深入某一类优化，可以沿着 FlashAttention、推测解码、prefix cache、调度和量化部署继续展开。
- [23. Speculative Decoding | 推测解码](./23_Speculative_Decoding.md)
- [24. SGLang RadixAttention | SGLang RadixAttention](./24_SGLang_RadixAttention.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [34. Prefix Caching and Chunked Prefill | Prefix Caching 与 Chunked Prefill](./34_Prefix_Caching_and_Chunked_Prefill.md)
- [35. Multi-Token Decoding | 多 Token 解码](./35_Multi_Token_Decoding.md)
- [36. Decode Scheduling | 解码调度](./36_Decode_Scheduling.md)
- [40. GPTQ and AWQ Weight Quantization | GPTQ 与 AWQ 权重量化](./40_GPTQ_and_AWQ_Weight_Quantization.md)
- [41. FP8 and KV Cache Quantization | FP8 与 KV Cache 量化](./41_FP8_and_KV_Cache_Quantization.md)
- [37. KV Cache Scheduling | KV Cache 调度](./37_KV_Cache_Scheduling.md)

---
### Step 1: 定义 workload 与固定 baseline
先回答一个问题：在同一模型、同一输入集和同一硬件环境下，哪种推理策略更划算？

- 固定模型、backend、batch size、prompt tokens、generated tokens、dtype、cache policy 和评测轮数。
- Baseline 建议从 `PyTorch eager + batch=1 + 固定 prompt/output length + warm-up + 多轮测量` 开始。
- 明确 candidate 只改一个变量，例如 FlashAttention、batch size、KV cache 策略、量化精度或推理后端。
- 统一核心指标：TTFT、TPOT、generated tokens/s、total latency、peak memory。
- 这节的目标不是证明某个方案“能跑”，而是在相同约束下输出可解释的推理选型结论。

### Step 2: 拆分 prefill / decode 并记录指标

推理项目不能只报一个总耗时。prefill 和 decode 的瓶颈不同，优化手段也不同。

- `prefill` 处理 prompt，通常和 prompt length、attention 计算、batch size、kernel 访存有关。
- `decode` 每步生成新 token，通常和 KV cache 读写、调度、batching、采样和小 batch 利用率有关。
- TTFT 可以近似看成 prefill latency，TPOT 可以看成 decode 阶段平均每 token 延迟。
- 同时记录 peak memory，避免只看吞吐而忽略 KV cache 或量化带来的显存变化。

### Step 3: 诊断瓶颈并选择候选方案

先判断项目主要受什么限制，再选择候选优化方向。

| 瓶颈类型 | 典型信号 | 候选方向 |
| --- | --- | --- |
| prefill-bound | prefill 占比高，长 prompt 变慢明显 | FlashAttention、chunked prefill、batching |
| decode-bound | TPOT 高，decode 占比高 | speculative decoding、multi-token decoding、decode scheduling |
| memory-bound | peak memory 接近预算，batch 上不去 | KV cache quantization、PagedAttention、GQA/MQA |
| balanced | 各项都不突出 | 保持 baseline 或做小步 profiling |

### Step 4: 输出推理选型报告

最后把 baseline 和 candidate 放到同一个口径下比较，输出可执行结论。

- 输出 baseline vs candidate 对比表，至少包含 TTFT、TPOT、throughput、total latency、peak memory 和瓶颈判断。
- 如果 candidate 只提升吞吐但明显拉高 TTFT，要说明适合离线批处理还是在线交互。
- 如果 candidate 显存更省但 TPOT 变差，要说明是否为了更大 batch 或更长上下文让路。
- 最终决策使用 `keep / tune / switch`：保留 baseline、继续调候选方案、或切换到 candidate。
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
candidate run ─► candidate comparison ───────────► keep / tune / switch
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| Workload | backend、batch、prompt tokens、generated tokens、dtype、cache policy | 保证可复现 |
| 指标 | TTFT、TPOT、throughput、total latency、peak memory | 保证同口径比较 |
| 诊断 | prefill-bound、decode-bound、memory-bound、balanced | 解释为什么优化有效或无效 |
| 对比 | latency / throughput / memory delta | 判断 candidate 是否值得保留 |
| 决策 | keep / tune / switch | 输出推理选型结论 |


```python
import time

```


```python
# 补全推理性能对比的六个关键函数
# 目标：完成 workload -> metrics -> bottleneck -> comparison -> decision 的最小项目链路

def build_inference_config(model_name, backend, batch_size, prompt_tokens, generated_tokens, dtype, cache_policy):
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
    # ==========================================
    # TODO 4: 诊断推理瓶颈
    # 规则：显存接近预算优先判 memory-bound；否则按 prefill/decode 占比判断。
    # ==========================================
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
    # ==========================================
    # TODO 6: 输出推理选型建议
    # 规则：吞吐明显提升且 TTFT 没明显退化则 switch；有收益但仍有瓶颈则 tune；否则 keep。
    # ==========================================
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
        assert decision['decision'] == 'switch', "吞吐提升且 TTFT 未明显退化时应建议 switch！"

        weak_comparison = dict(comparison)
        weak_comparison['throughput_gain'] = 0.02
        weak_comparison['ttft_delta_ms'] = 1.0
        assert recommend_inference_decision(weak_comparison, decode_bound)['decision'] == 'tune', "小幅收益但仍有瓶颈时应建议 tune！"

        bad_comparison = dict(comparison)
        bad_comparison['throughput_gain'] = -0.05
        bad_comparison['ttft_delta_ms'] = -30.0
        assert recommend_inference_decision(bad_comparison, {'bottleneck': 'balanced'})['decision'] == 'keep', "没有收益且 TTFT 退化时应建议 keep！"

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
        decision = 'switch'
        reason = 'candidate 吞吐提升明显，TTFT 退化在可接受范围内，可以切换到候选方案。'
    elif comparison['throughput_gain'] > 0.0 and candidate_bottleneck['bottleneck'] != 'balanced':
        decision = 'tune'
        reason = 'candidate 已有收益，但瓶颈仍然存在，继续围绕诊断结果调参或换策略。'
    else:
        decision = 'keep'
        reason = 'candidate 收益不足或交互延迟退化明显，先保留 baseline。'
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
- **switch**：吞吐提升达标，TTFT 退化在可接受范围内。
- **tune**：candidate 有收益，但瓶颈仍然存在，需要继续沿诊断方向调参。
- **keep**：candidate 收益不足，或交互延迟退化明显，先保留 baseline。
- **项目意义**：推理选型不能只看一个指标。最终结论要同时考虑 workload、吞吐、TTFT、TPOT、显存和瓶颈类型。

**推理性能对比的实验原则**
- **变量控制**：同一轮对比中只改一个变量，例如 batch size、precision、推理后端或 cache 策略。
- **指标闭环**：每次实验至少记录 TTFT、TPOT、throughput 和 peak memory。
- **阶段拆分**：把 prefill 和 decode 分开看，避免把长 prompt 问题误判成 decode 问题。
- **结果复盘**：最终输出要回扣 Step 1 的问题：在给定约束下，哪种推理策略最划算，理由是什么。
