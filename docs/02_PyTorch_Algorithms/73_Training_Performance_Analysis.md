# 73. Training Performance Analysis | 训练性能分析

**难度：** Hard | **环境：** CPU-first | **标签：** `显存优化`, `训练剖析`, `性能分析` | **目标人群：** 项目决策练习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

这一节对应的真实项目问题不是“训练为什么慢”，而是“在既定训练任务、收敛约束和显存预算下，这次性能改动是否值得保留”。真实工程里，读者真正要判断的不是单独的 step time，而是 baseline 与 tuned 方案在数据加载、forward / backward、optimizer step 和 peak memory 拆开之后，是否还能支撑可解释的训练优化结论。

本节的核心矛盾是训练速度、显存占用与收敛稳定性之间的权衡：优化可以让 step 更快、samples/s 更高，也可能因为显存峰值、同步等待或 loss 变差而把收益抵消掉。做完这一节，你应该能输出一份 baseline vs tuned 的训练性能分析结论，而不只是记录一组总耗时数字。

因此，这一页把训练性能分析收成一个最小项目交付入口：先定义训练性能分析目标，再拆解 training step、定位瓶颈、做单点复测，并把结论收成一份可复用的训练性能报告。它直接承接 `09 / 17 / 19 / P1:13` 的训练与显存优化直觉，并继续通向 `74` 的端到端优化闭环和 `60` 的 LoRA 微调项目。

如果把它放回显存优化路线，一个更直接的读法是：这页不只是泛泛地比较训练快慢，而是优先帮助你判断 `checkpointing / offload / mixed precision / batch` 这些训练侧显存手段到底把峰值显存压下去了多少，又把 step time 和稳定性拉坏了多少。也就是说，它在显存路线里承担的是“证据链页”，不是另一个纯机制页。

**关键词：** `training`, `profiling`, `memory`, `step time`

---

## 前置阅读

**导语：** 先把训练循环、反向传播和显存优化的最小口径理顺，再进入训练性能分析；本节重点不是重复讲机制，而是把训练瓶颈放到同一张分析表里比较。
- [09. SFT Training Loop | SFT 训练循环](./09_SFT_Training_Loop.md)
- [17. Autograd Basics | Autograd 基础](./17_Autograd_Basics.md)
- [19. Activation Checkpointing and Activation Offload | 激活检查点与激活卸载](./19_Activation_Checkpointing_and_Activation_Offload.md)
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)

## 相关阅读

**导语：** 做完训练性能分析后，最自然的下一步是把瓶颈定位推进到端到端优化闭环，或回到 LoRA 项目核对训练成本是否真的划算；如果你正在走显存路线，重点要看这些证据是否支撑 `checkpointing / offload / mixed precision / batch` 的取舍。
- [74. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](./60_LoRA_Fine_Tuning_Project.md)
---
### Step 1: 定义训练性能分析目标

- 固定模型、输入数据、batch size、seq len、硬件环境和运行后端，保证 baseline 与 tuned 只差一个变量。
- 明确优化目标，例如降低 step time、提升 samples/s、降低 peak memory 或减少同步等待。
- 同时写清约束条件：训练任务要保留 loss / 收敛约束，不能只追求单项速度收益。
- Baseline 需要能稳定复现，不能只跑一次；建议先 warm-up，再测多轮平均值。
- 这一步的目标是让后面的性能分析有判断标准，而不是只得到一组孤立数字。

### Step 2: 先拆解 training step 并确认 baseline 合法

训练性能分析必须先确认 baseline 可复现，再把一个 step 拆成能归因的几段，不能直接对着总耗时盲目调参。

- 数据加载：DataLoader、CPU 预处理、CPU -> GPU 拷贝是否让 GPU 等待。
- 前向计算：Attention、Linear、LayerNorm 等 forward kernel 是否占主要时间。
- 反向计算：backward kernel、梯度计算和梯度累积是否成为瓶颈。
- 优化器更新：optimizer step 是否占用明显时间。
- 显存峰值：激活、梯度、优化器状态和临时 buffer 是否接近上限。
- 同步开销：是否存在不必要的 CPU/GPU 同步或多卡通信等待。

这一步的目标是把“训练慢”具体化成某一类瓶颈，而不是只得到一个模糊结论。
### Step 3: 用统一口径比较收益与代价

训练优化项目必须同时看 step time、samples/s、peak memory 和 loss，不能只挑单项速度收益下结论。

- 一次只改一个变量，例如 batch size、混合精度、gradient checkpointing、数据加载或同步点。
- 改完后重新测同样的指标，比较 baseline / tuned 的差异。
- 如果 step time 变快但 loss 异常、显存更高或稳定性变差，要把取舍写清楚。
- 这一轮修改的目标是建立因果关系，而不是一次性把所有开关都打开。

这一步的目标是回答：这次改动是把瓶颈解决了，还是只是把瓶颈挪走了。
### Step 4: 输出训练性能结论

训练性能分析最终不是输出“总耗时有没有降”，而是输出这次改动在当前训练任务下是否值得继续保留、微调或回退。

- 输出 baseline / tuned 对比表，至少包含 step time、samples/s、peak memory、loss 和备注。
- 附上 profiling 截图或关键统计，说明瓶颈来自数据、forward、backward、optimizer 还是显存。
- 写清楚本次改动、收益、代价和是否满足 loss / 收敛约束。
- 如果还有后续优化空间，就列出下一轮优先级。

这一步的目标是把训练性能分析收成一份可复用的项目报告。
### Step 5: 最小代码模板

上面的 Step 1-4 是完整训练性能分析流程。下面的代码只实现其中最小、可复用的三块：测量训练 step 的平均耗时与峰值显存、汇总 baseline / tuned 的差异，以及把结果收成 `accept / tune / reject` 的轻量项目决策。真实项目中的 forward / backward / optimizer 拆解和 loss 约束，需要在 profiling 报告中继续补充。

### 提示

- 先固定 baseline，再看 tuned，不要把环境变量、batch、seq len 一起改掉。
- GPU 场景下要关注 peak memory，CPU 场景下至少要保证计时口径一致。
- 一次只改一个变量，才能把 step time 和显存变化归因到具体修改。
### 测试

运行下面的测试单元，确认 `measure_train_step` 和 `summarize_training_result` 的输出字段完整且口径一致。

```python
import time
import torch

```


```python
# 完成训练性能统计的两个函数
# 目标：完成 measure -> compare 的最小训练性能分析链路

def measure_train_step(train_step_fn, warmup=2, iters=8):
    # ==========================================
    # TODO 1: 记录平均 step time 和 peak memory
    # 提示：先 warmup，再测正式迭代；GPU 场景下记录 peak memory。
    # ==========================================
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # start = ???
    for _ in range(iters):
        train_step_fn()
    # end = ???
    # elapsed = ???

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        # peak_mem_mb = ???
        pass

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

def summarize_training_result(base_metrics, tuned_metrics):
    # ==========================================
    # TODO 2: 比较 baseline 和 tuned 的指标差值
    # 提示：delta = baseline - tuned，正数表示 tuned 更省或更快。
    # ==========================================
    # time_delta = ???
    # mem_delta = ???
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
    """根据速度和显存收益给出训练项目结论。"""
    # ==========================================
    # TODO 3: 输出训练项目结论
    # 规则：
    # - 速度和显存收益都达标：accept
    # - 至少一项有正收益：tune
    # - 否则：reject
    # ==========================================
    # strong_time_gain = ???
    # strong_memory_gain = ???
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
    #     decision = ???
    #     reason = ???
    # else:
    #     decision = ???
    #     reason = ???
    # return {'decision': decision, 'reason': reason}

```


```python
# 测试你的实现
def test_training_project_template():
    try:
        counter = {'n': 0}

        def train_step():
            counter['n'] += 1

        result = measure_train_step(train_step, warmup=0, iters=2)
        assert counter['n'] == 2, "measure_train_step 没有正确执行训练迭代次数！"
        assert 'step_time_ms' in result and 'peak_mem_mb' in result, "训练统计字段不完整！"
        assert result['step_time_ms'] >= 0.0, "step_time_ms 应为非负数！"
        assert result['peak_mem_mb'] >= 0.0, "peak_mem_mb 应为非负数！"

        baseline = {'step_time_ms': 120.0, 'peak_mem_mb': 8192.0}
        tuned = {'step_time_ms': 98.0, 'peak_mem_mb': 6144.0}
        summary = summarize_training_result(baseline, tuned)
        assert summary['step_time_delta_ms'] == 22.0, "step_time_delta_ms 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 2048.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['time_improved'] is True, "time_improved 判断不正确！"
        assert summary['memory_improved'] is True, "memory_improved 判断不正确！"
        decision = recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)
        assert decision['decision'] == 'accept', "速度和显存收益都达标时应建议 accept！"

        weak_summary = {'step_time_delta_ms': 6.0, 'peak_mem_delta_mb': 256.0, 'time_improved': True, 'memory_improved': True}
        assert recommend_training_decision(weak_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)['decision'] == 'tune', "收益不够稳时应建议 tune！"

        bad_summary = {'step_time_delta_ms': -4.0, 'peak_mem_delta_mb': 0.0, 'time_improved': False, 'memory_improved': False}
        assert recommend_training_decision(bad_summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0)['decision'] == 'reject', "没有形成有效收益时应建议 reject！"
        print("✅ 训练性能分析项目模板代码通过基础校验。")

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


test_training_project_template()

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
import time
import torch

# TODO 1: 测量训练 step 的平均耗时和峰值显存
def measure_train_step(train_step_fn, warmup=2, iters=8):
    for _ in range(warmup):
        train_step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        train_step_fn()
    end = time.perf_counter()
    elapsed = (end - start) / iters

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        'step_time_ms': round(elapsed * 1000, 2),
        'peak_mem_mb': round(peak_mem_mb, 2),
    }

# TODO 2: 汇总 baseline 和 tuned 的差异
def summarize_training_result(base_metrics, tuned_metrics):
    time_delta = base_metrics['step_time_ms'] - tuned_metrics['step_time_ms']
    mem_delta = base_metrics['peak_mem_mb'] - tuned_metrics['peak_mem_mb']
    return {
        'step_time_delta_ms': round(time_delta, 2),
        'peak_mem_delta_mb': round(mem_delta, 2),
        'time_improved': time_delta > 0,
        'memory_improved': mem_delta > 0,
    }

# TODO 3: 输出训练项目结论
def recommend_training_decision(summary, min_time_delta_ms=10.0, min_memory_delta_mb=512.0):
    strong_time_gain = summary['step_time_delta_ms'] >= min_time_delta_ms
    strong_memory_gain = summary['peak_mem_delta_mb'] >= min_memory_delta_mb
    if strong_time_gain and strong_memory_gain:
        decision = 'accept'
        reason = '训练速度和显存收益都达标，值得继续保留当前优化。'
    elif summary['time_improved'] or summary['memory_improved']:
        decision = 'tune'
        reason = '至少有一项收益成立，但还没形成稳定项目结论，先继续微调。'
    else:
        decision = 'reject'
        reason = '速度和显存都没有形成有效收益，当前改动不值得保留。'
    return {'decision': decision, 'reason': reason}

counter = {'n': 0}
def train_step():
    counter['n'] += 1
print(measure_train_step(train_step, warmup=0, iters=2))

```

### 解析

**1. TODO 1: 统计训练 step 耗时和峰值显存**
- **实现方式**：先执行 `warmup` 轮训练 step 预热，再用 `time.perf_counter()` 记录正式测量阶段的起点和终点，最后用 `(end - start) / iters` 得到平均 step time。
- **关键点**：warmup 不计入结果，避免首次运行的数据加载、kernel 初始化或缓存状态影响平均耗时。
- **显存统计**：GPU 场景下先调用 `torch.cuda.reset_peak_memory_stats()` 清空历史峰值，再用 `torch.cuda.max_memory_allocated()` 读取本轮训练的峰值显存。CPU 场景下返回 `0.0`，保证模板可以在无 GPU 环境中运行。

**2. TODO 2: 汇总 baseline 和 tuned 的差异**
- **实现方式**：`time_delta = baseline_step_time - tuned_step_time`，`mem_delta = baseline_peak_mem - tuned_peak_mem`。
- **关键点**：这里统一用 `baseline - tuned`，所以 delta 为正表示优化后更快或更省显存。
- **技术细节**：`time_improved` 和 `memory_improved` 只是快速判断标记，真正复盘时还要结合 loss、吞吐和收敛稳定性一起看。

**3. TODO 3: 输出训练项目结论**
- **accept**：速度和显存收益都达标，说明当前改动值得保留并继续推进。
- **tune**：至少有一项收益成立，但还没达到稳定项目结论，适合继续围绕当前方向微调。
- **reject**：速度和显存都没有形成有效收益，说明当前改动不值得继续保留。

**训练性能分析的实验原则**
- **固定 baseline**：同一轮对比中固定模型、数据、batch size、seq len、优化器和评测方式。
- **一次只改一个变量**：例如只改 batch size、混合精度、gradient checkpointing 或数据加载方式，避免结果不可归因。
- **指标一起看**：step time 变快但 peak memory、loss 或稳定性变差时，要把取舍写清楚。
- **瓶颈归因**：如果 step time 没有改善，需要回到 profiling 结果，判断瓶颈来自数据等待、前向 / 反向算子，还是显存压力。
- **工程产物**：建议保存对比表、profiling 截图、瓶颈结论和下一轮计划，形成可复用的训练性能排障记录。
