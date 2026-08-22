# 74. Profiling Driven End to End Optimization | profiling 驱动的端到端优化

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `性能剖析`, `端到端优化` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

本节练习如何把一次性能优化整理成可复现的 profiling 闭环。你需要完成 baseline 测量、瓶颈定位、优化前后复测和结果汇总，并保证每次测量使用相同 workload 与指标口径。最终提交一份简短报告：哪里慢、为什么慢、改了什么、提升了多少，以及是否值得继续投入。
**层级定位：** 本项目是跨层 profiling 项目，覆盖 L1-L4 的执行证据；若采集资源调度、版本发布或服务可用性指标，才延伸到 L5。它负责定位和验证，不替代具体的算子优化、推理服务或平台治理项目。

**关键词：** `profiling`, `optimization`, `end-to-end`

---

## 前置阅读

**导语：** 先把 profiling 方法、训练/推理对比项目和训练性能分析看过，再进入端到端优化闭环会更容易判断改动是否有效。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](./60_LoRA_Fine_Tuning_Project.md)

## 相关阅读

**导语：** 完成优化闭环后，可以继续把瓶颈定位推进到更底层的实现手段，或回到并行/系统项目页验证是否值得迁移。
- [79. Distributed Parallel Benchmark | 分布式并行基准项目](./79_Distributed_Parallel_Benchmark.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
---
### Step 1: 定义端到端优化目标
先回答一个问题：这次优化到底要解决什么瓶颈，成功标准是什么？

- 固定模型、输入数据、batch size、seq len、硬件环境和运行后端，保证对比对象只差一个变量。
- 明确优化目标，例如降低 step time、提升 throughput、降低 peak memory 或减少通信等待。
- 同时写清约束条件：训练任务要保留 loss / accuracy 约束，推理任务要保留精度、输出一致性或服务 SLA 约束。
- Baseline 需要能稳定复现，不能只跑一次；建议至少 warm-up 若干轮，再测多轮平均值。
- 这一步的目标是让后面的优化有判断标准，而不是只得到一组孤立数字。

### Step 2: 先确认 baseline 和 profiling 口径合法

profiling 优化必须先确认 baseline 可复现，再把“慢”拆成可解释的瓶颈类型，不能直接对着单次热点截图开刀。

- 推荐先记录总耗时、吞吐、峰值显存，再用 profiler 看热点算子和同步点。
- 训练场景优先拆成：数据加载、forward、backward、optimizer step、显存峰值和多卡通信。
- 推理场景优先拆成：prefill、decode、KV cache、采样逻辑、数据搬运和 kernel 开销。
- 不要只找“最慢的一行代码”，而要判断瓶颈属于哪一类资源：计算、显存容量、内存带宽、通信还是调度。
- 这一步的产物应该是一句话瓶颈结论，例如：`当前瓶颈主要来自 decode 阶段 KV cache 读取`。

### Step 3: 用统一口径比较收益与代价

profiling 项目必须同时看 step time、throughput、peak memory 和任务约束，不能只挑单项热点收益下结论。

- 一次只改一个方向，例如调整 batch size、开启混合精度、替换 kernel、减少同步点或改变 cache 策略。
- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果改动影响训练 loss、推理输出、显存峰值或系统稳定性，要把代价写清楚。
- 如果某个改动只是在一项指标上变好，却让另一项变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是一次性把所有优化开关都打开。

### Step 4: 输出端到端优化结论

端到端优化最终不是输出“某个热点是不是降了”，而是输出这次改动在当前任务约束下是否值得继续保留、微调或回退。

- 输出 baseline / tuned 对比表，至少包含 step time、throughput、peak memory 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自哪一类资源。
- 写清楚本次改动、收益、代价和是否满足约束。
- 如果优化没有达到目标，记录失败原因和下一轮优先级。
- 最终产物应回答：原始瓶颈是什么，做了什么改动，收益有多大，这个改动是否值得保留。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整 profiling 驱动优化流程。下面的代码实现其中最小、可复用的四块：测平均耗时、汇总 baseline / tuned 指标差异、生成优化报告，以及把结果收成 `accept / tune / reject` 的轻量决策。真实项目中的 profiling 截图、瓶颈证据和优化策略，需要基于这四步继续补充。

### 提示

- 先固定 baseline，再做 profiling，再改一个变量。
- 不要只看 step time，至少同时记录 throughput 和 peak memory。
- 如果瓶颈不是单一算子，而是同步 / 调度 / cache，报告里要直接写出来。

```python
import time

```


```python
def benchmark_fn(fn, warmup=3, iters=10):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    # start = ???
    for _ in range(iters):
        fn()
    # total = ???
    # avg_time_ms = ???
    return avg_time_ms

def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    # time_delta = ???
    # memory_delta = ???
    # throughput_delta = ???

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    # rows = ???
    # conclusion = ???
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    # ==========================================
    # TODO 4: 给出轻量优化决策
    # 规则：
    # - 时间和吞吐都改善：accept
    # - 时间改善，且显存或吞吐至少有一项为正收益：tune
    # - 否则：reject
    # ==========================================
    # strong_time_gain = ???
    # strong_memory_gain = ???
    # strong_throughput_gain = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # positive_memory_gain = summary['peak_mem_delta_mb'] > 0
    # positive_throughput_gain = summary['throughput_delta'] > 0
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    # return {'decision': decision, 'reason': reason}
    raise NotImplementedError

```

### 测试


```python
def test_optimization_project_template():
    try:
        counter = {'n': 0}

        def fn():
            counter['n'] += 1

        result = benchmark_fn(fn, warmup=0, iters=2)
        assert counter['n'] == 2, "benchmark 应该运行 iters 次"
        assert result >= 0.0, "平均耗时应该非负"

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0, 'throughput': 80.0}
        tuned = {'step_time_ms': 96.0, 'peak_mem_mb': 7168.0, 'throughput': 100.0}
        summary = summarize_optimization_result(baseline, tuned)

        assert summary['step_time_delta_ms'] == 24.0
        assert summary['peak_mem_delta_mb'] == 1024.0
        assert summary['throughput_delta'] == 20.0
        assert summary['time_improved'] is True
        assert summary['memory_improved'] is True
        assert summary['throughput_improved'] is True

        report = format_optimization_report(summary, 'backward kernel 占比过高', '保留混合精度并继续检查 optimizer')
        assert '| 指标 | 变化 | 判断 |' in report
        assert 'backward kernel 占比过高' in report
        assert '保留混合精度并继续检查 optimizer' in report

        decision = recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert decision['decision'] == 'accept'

        mixed_summary = {'step_time_delta_ms': 12.0, 'peak_mem_delta_mb': 128.0, 'throughput_delta': 2.0, 'time_improved': True, 'memory_improved': True, 'throughput_improved': True}
        mixed_decision = recommend_optimization_decision(mixed_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert mixed_decision['decision'] == 'tune'

        weak_summary = {'step_time_delta_ms': -3.0, 'peak_mem_delta_mb': 64.0, 'throughput_delta': 1.0, 'time_improved': False, 'memory_improved': True, 'throughput_improved': True}
        weak_decision = recommend_optimization_decision(weak_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0)
        assert weak_decision['decision'] == 'reject'

        print("✅ profiling 驱动的端到端优化项目模板代码通过基础校验。")
    except NotImplementedError:
        print("请先完成 TODO 代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义")
        raise NotImplementedError("请先完成 TODO 代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 代码！") from e


test_optimization_project_template()

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
def benchmark_fn(fn, warmup=3, iters=10):
    # ==========================================
    # TODO 1: 先做 warmup，再测量平均耗时
    # 提示: 用 time.perf_counter() 记录起止时间
    # 返回单位统一为 ms，方便和项目报告对齐
    # ==========================================
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - start
    avg_time_ms = total / iters * 1000
    return avg_time_ms


def summarize_optimization_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 汇总 baseline / tuned 的核心指标差异
    # 提示: 正数表示 tuned 相比 baseline 有改善
    # step time / memory 越低越好，throughput 越高越好
    # ==========================================
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    memory_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    throughput_delta = tuned_metrics['throughput'] - base_metrics['throughput']

    summary = {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'throughput_delta': round(throughput_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': memory_delta > 0,
        'throughput_improved': throughput_delta > 0,
    }
    return summary


def format_optimization_report(summary, bottleneck, next_action):
    # ==========================================
    # TODO 3: 生成一段最小优化报告
    # 提示: 把指标变化、瓶颈结论和下一步动作放在一起
    # ==========================================
    header = "| 指标 | 变化 | 判断 |"
    sep = "| --- | --- | --- |"
    rows = [
        f"| step time | {summary['step_time_delta_ms']} ms | {'改善' if summary['time_improved'] else '未改善'} |",
        f"| peak memory | {summary['peak_mem_delta_mb']} MB | {'改善' if summary['memory_improved'] else '未改善'} |",
        f"| throughput | {summary['throughput_delta']} | {'改善' if summary['throughput_improved'] else '未改善'} |",
    ]
    conclusion = f"瓶颈判断：{bottleneck}。下一步：{next_action}。"
    return "\n".join([header, sep] + rows + [conclusion])


def recommend_optimization_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0, min_throughput_delta=5.0):
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    strong_throughput_gain = summary['throughput_delta'] >= min_throughput_delta
    positive_memory_gain = summary['peak_mem_delta_mb'] > 0
    positive_throughput_gain = summary['throughput_delta'] > 0
    if strong_time_gain and strong_throughput_gain:
        decision = 'accept'
        reason = '时间与吞吐改善都达标，当前优化值得保留。'
    elif strong_time_gain and (positive_memory_gain or positive_throughput_gain):
        decision = 'tune'
        reason = '时间改善成立，但资源或吞吐收益还不够稳，建议继续微调。'
    else:
        decision = 'reject'
        reason = '当前优化没有形成稳定的端到端收益，建议回退或重新定位瓶颈。'
    return {'decision': decision, 'reason': reason}

```

### 解析

这一版题目区保留 `4` 个核心 TODO：测量、汇总、报告和轻量决策。这里不把 profiling 页做成重型项目审计器，而是让读者先掌握 `measure -> summarize -> report -> decide` 的最小项目闭环。

- **这一题要解决什么**：把 profiling 优化流程压缩成一个最小可复用模板，保证每次优化都能留下可比较的指标和明确结论。
- **为什么这样做**：性能优化不能只看单次运行结果，必须固定 baseline、测量同一组指标，并把改动前后的差异收敛成项目报告。
- **带走的直觉**：profiling 的价值不是“找到一个慢点”，而是建立 `测量 -> 定位 -> 修改 -> 复测 -> 复盘` 的闭环。

**1. TODO 1 (benchmark_fn)**

- **warmup**：先运行若干轮，不计入统计，避免初始化、缓存和调度抖动影响结果。
- **计时范围**：只把正式测量的 `iters` 轮放进 `start / total` 之间。
- **单位统一**：返回 ms，而不是秒，方便和 step time、latency 表格放在一起比较。
- **工程注意**：真实 GPU 场景下通常还需要 `torch.cuda.synchronize()`，否则异步 kernel 可能让计时偏小；本节保持 CPU-first，不强制加入。

**2. TODO 2 (summarize_optimization_result)**

- **step time 差值**：`baseline - tuned`，正数表示 tuned 更快。
- **peak memory 差值**：`baseline - tuned`，正数表示 tuned 更省显存。
- **throughput 差值**：`tuned - baseline`，正数表示 tuned 吞吐更高。
- **布尔判断**：`time_improved / memory_improved / throughput_improved` 把数值变化变成可读结论，方便项目报告直接引用。

**3. TODO 3 (format_optimization_report)**

- **表格部分**：把核心指标变化放到同一张 Markdown 表里，便于复盘和横向比较。
- **瓶颈判断**：不要只输出数字，还要写清楚瓶颈来自哪里，例如数据加载、backward kernel、KV cache 或通信同步。
- **下一步动作**：每轮优化结束都应该留下后续优先级，否则下一轮很容易重新从零开始定位。

**4. TODO 4 (recommend_optimization_decision)**

- **accept**：时间改善和吞吐改善都达标，说明这次改动对端到端目标确实有帮助。
- **tune**：时间改善成立，但显存或吞吐收益还不够稳，说明这次优化方向可能对，但还没到可直接保留的程度。
- **reject**：没有形成稳定的端到端收益，应该回退或重新定位瓶颈，而不是继续堆优化开关。

**项目化原则**

- **一次只改一个变量**：否则收益不可归因。
- **指标要成组出现**：只看变快不够，还要看显存、吞吐、loss / 精度或输出一致性。
- **结论要回扣目标**：最终判断必须回答 Step 1 的问题：这次优化是否达成目标，是否值得保留。
