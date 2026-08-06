# 60. LoRA Fine Tuning Project | LoRA 微调项目

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `LoRA`, `Finetuning` | **目标人群：** 模型微调与工程部署

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面已经分别讲过 SFT 数据构造、LoRA 机制、学习率调度、梯度累积和端到端训练报告，但真实项目里不能只回答“LoRA 能不能跑”。更关键的问题是：数据是否干净，loss mask 是否正确，LoRA 相比 baseline 到底少训练了多少参数，省了多少显存，速度和 train/val loss 又付出了什么代价。

本节把 LoRA 微调做成一个项目交付模板：训练前先做数据审计和 loss mask 抽样核对，中间固定 LoRA 配置、训练口径和参数账本，训练后再把 adapter artifact、merge 检查、sanity generation、资源指标和采用建议收成 baseline vs LoRA 的项目结论。代码区只实现最小可复用的项目配置、数据审计、loss 核对、参数账本、结果汇总和交付检查，完整训练循环、loss 曲线和 profiling 截图可以基于这份报告继续补充。

**关键词：** `LoRA`, `training`, `project`, `profiling`, `report`

---
## 前置阅读

**导语：** 先看 LoRA 机制、端到端训练闭环和显存优化，再做这个项目；本节默认你已经知道训练循环怎么跑，重点转向 LoRA 方案是否值得采用。
- [10. LoRA Tutorial | LoRA 教程](./10_LoRA_Tutorial.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](./11_LR_Schedulers_WSD_Cosine.md)
- [12. Gradient Accumulation | 梯度累积](./12_Gradient_Accumulation.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](./13_End_to_End_Fine_Tuning_Experiment.md)
- [P0: 20. Profiling and Memory Ledger | 性能剖析与显存账本](../00_Prerequisites/20_Profiling_and_Memory_Ledger.md)

## 相关阅读

**导语：** 完成 LoRA 项目账本后，建议继续用训练性能分析、推理性能对比和 profiling 方法验证这套方案的实际成本。
- [P1: 13. Profiling and Bottleneck Analysis | 性能分析与瓶颈定位](../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.md)
- [P1: 19. Operator Fusion Introduction | 算子融合导论](../01_Hardware_Math_and_Systems/19_Operator_Fusion_Introduction.md)
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [73. Training Performance Analysis | 训练性能分析](./73_Training_Performance_Analysis.md)


### Step 1: 定义 LoRA 微调目标
先回答一个问题：在尽量少训练参数的前提下，LoRA 能否完成目标任务，并保留可接受的 train / val loss 表现？

- 固定底座模型、数据集、batch size、seq len、优化器、学习率和训练 step 数。
- 明确 baseline 是全参数微调、冻结底座不训练，还是已有的普通微调配置。
- 训练前先做数据审计：样本数、空 response、重复样本、超长样本和长度分布。
- 抽样核对 `input_ids / attention_mask / labels`：response 是否进入 loss，padding 是否被 `-100` 屏蔽。
- 记录 LoRA 配置：target modules、rank、alpha、dropout、learning rate、micro batch、accum steps 和 scheduler。
- 统一记录核心指标：可训练参数量、参数占比、step time、peak memory、train loss、val loss。
- 这节先建立 LoRA 项目交付模板，再把数据、loss、参数、显存、速度、效果和 artifact 收成一份项目汇总。

### Step 2: 跑通 baseline 并记录账本

LoRA 的收益必须和稳定 baseline 对比，不能只看 LoRA 自己能不能跑。

- 先在同一批样本和同一套训练配置下跑通 baseline。
- 记录 baseline 的可训练参数量、train/val loss、平均 step time 和 peak memory。
- 确认 baseline loss 能正常下降，再进入 LoRA 对比。
- 如果 baseline 本身不稳定，后面的 LoRA 结果就没有可解释性。

### Step 3: 插入 LoRA 并做同口径对比

把 LoRA adapter 插到 attention projection 或 MLP linear layer 上，只训练低秩旁路。

- 冻结底座权重，只让 LoRA 的 `A / B` 矩阵参与训练。
- 先计算单层 LoRA 参数量，再估算多层插入后的总可训练参数量。
- 用同样的 batch、输入长度、训练步数和评估方式比较 LoRA 与 baseline。
- 重点看三个问题：参数量省了多少，显存 / 速度是否改善，train/val loss 是否仍然正常。

### Step 4: 输出微调项目结论

最后把 LoRA 和 baseline 放到同一张表里，说明这次微调方案是否值得采用。

- 输出 baseline vs LoRA 对比表，至少包含 trainable params、param ratio、step time、peak memory、train loss、val loss。
- 写清楚 LoRA 节省的是训练参数和优化器状态，不等于底座模型权重不存在。
- 记录本次 target modules、rank、alpha、dropout、学习率、effective batch 和 scheduler，方便后续复现实验。
- 保存 adapter，并记录 tokenizer、special tokens、merge 检查和最小生成样例检查。
- 如果效果不足，下一轮优先调整 rank、插层范围、学习率或 gradient accumulation。
- 最终产物应回答：数据和 loss 是否可信，LoRA 少训练了多少参数，换来了多少显存 / 速度收益，val loss 损失是否还能接受，adapter 是否可以交付。

### Step 5: 最小代码模板

上面的 Step 1-4 是完整 LoRA 微调项目流程。下面的代码实现其中最小、可复用的六块：数据审计、loss mask 核对、项目配置、LoRA 参数账本、结果汇总和交付检查。
#### 图解：09-13 如何收束到 LoRA 项目报告

`60` 不重复实现训练循环，而是把前面几节已经跑通的机制收成一份可复现的项目报告。

```text
09 SFT data       input_ids / attention_mask / labels
      │
10 LoRA           target modules / rank / alpha / dropout
      │
11 Scheduler      lr schedule counted by optimizer update
      │
12 Accumulation   micro batch -> effective batch
      │
13 E2E report     initial/final train loss + val loss
      │
      ▼
60 LoRA project   data audit + loss mask + parameter ledger + artifacts + decision
```

项目页最小产物：

| 模块 | 必须记录 | 用途 |
|:---|:---|:---|
| 数据 | 样本数、空 response、重复样本、超长样本 | 证明训练输入可信 |
| Loss | supervised tokens、padding supervised tokens | 证明 loss 口径正确 |
| 配置 | target modules、rank、alpha、dropout、lr、effective batch | 保证可复现 |
| 账本 | trainable params、param ratio | 证明 LoRA 是否省参数 |
| 训练结果 | train/val loss、step time、peak memory | 判断效果和成本 |
| 交付 | adapter、tokenizer、merge check、sanity generation | 判断是否能交付 |
| 决策 | accept / tune / reject | 输出项目结论 |

#### 图解：微调项目 v2 的交付链路

```text
training data ──► data audit ──► loss mask check ──► baseline run
                                                        │
                                                        ▼
LoRA config ──► adapter training ──► metric comparison ──► artifact check ──► final decision
```


```python
import math

```


```python
import math

# TODO: 完成 LoRA 项目配置、数据审计、loss 核对、参数账本和项目汇总
# 目标：从 09-13 的训练闭环收束到 baseline vs LoRA 项目交付报告

def audit_sft_examples(examples, max_total_chars):
    # ==========================================
    # TODO 1: 审计 SFT 样本
    # 提示：检查样本数、空 response、重复 prompt/response 和超长样本。
    # ==========================================
    # total_samples = ???
    # empty_response_count = ???
    # duplicate_count = ???
    # over_length_count = ???
    # avg_total_chars = ???
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_total_chars': round(avg_total_chars, 2),
    }

def loss_mask_report(attention_mask, labels, ignore_index=-100):
    # ==========================================
    # TODO 2: 核对 loss mask
    # 提示：labels != -100 的 token 会参与 loss；attention_mask == 0 的 padding 不应参与 loss。
    # ==========================================
    # total_tokens = ???
    # non_padding_tokens = ???
    # supervised_tokens = ???
    # padding_supervised_tokens = ???
    # supervised_ratio = ???
    return {
        'total_tokens': total_tokens,
        'non_padding_tokens': non_padding_tokens,
        'supervised_tokens': supervised_tokens,
        'padding_supervised_tokens': padding_supervised_tokens,
        'supervised_ratio': round(supervised_ratio, 4),
    }

def build_lora_project_config(
    base_model,
    target_modules,
    rank,
    alpha,
    dropout,
    learning_rate,
    micro_batch_size,
    accum_steps,
    scheduler,
):
    # ==========================================
    # TODO 3: 汇总 LoRA 项目配置
    # 提示：effective_batch_size = micro_batch_size * accum_steps
    # ==========================================
    # effective_batch_size = ???
    return {
        'base_model': base_model,
        'target_modules': target_modules,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'micro_batch_size': micro_batch_size,
        'accum_steps': accum_steps,
        'effective_batch_size': effective_batch_size,
        'scheduler': scheduler,
    }

def lora_trainable_params(in_dim, out_dim, rank):
    # ==========================================
    # TODO 4: 计算单层 LoRA 的可训练参数量
    # 提示：LoRA 旁路包含 A 和 B 两个低秩矩阵。
    # ==========================================
    # trainable_params = ???
    return trainable_params

def full_linear_params(in_dim, out_dim):
    # ==========================================
    # TODO 5: 计算完整线性层的参数量
    # 提示：这里只统计 weight，不额外考虑 bias。
    # ==========================================
    # total_params = ???
    return total_params

def lora_param_ratio(in_dim, out_dim, rank):
    # ==========================================
    # TODO 6: 计算 LoRA 参数占比
    # 提示：先算 LoRA 参数量和全参基线，再做比例。
    # ==========================================
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    # ratio = ???
    return ratio

def summarize_lora_project(baseline_metrics, lora_metrics):
    # ==========================================
    # TODO 7: 汇总 baseline 和 LoRA 的项目指标
    # 提示：资源类 delta = baseline - lora；loss delta = lora - baseline。
    # ==========================================
    # param_reduction = ???
    # memory_delta = ???
    # time_delta = ???
    # train_loss_delta = ???
    # val_loss_delta = ???
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'final_train_loss_delta': round(train_loss_delta, 4),
        'final_val_loss_delta': round(val_loss_delta, 4),
    }

def build_adapter_artifact_record(adapter_path, tokenizer_path, merge_checked, sanity_generation_checked):
    # ==========================================
    # TODO 8: 记录 adapter 交付物
    # 提示：微调项目交付时至少要知道 adapter、tokenizer、merge check 和 sanity generation 是否完成。
    # ==========================================
    return {
        'adapter_path': adapter_path,
        'tokenizer_path': tokenizer_path,
        'merge_checked': merge_checked,
        'sanity_generation_checked': sanity_generation_checked,
    }

def check_lora_project_readiness(data_audit, mask_report, artifact_record):
    # ==========================================
    # TODO 9: 检查项目是否可以交付
    # 提示：数据、loss mask、adapter artifact 任一关键项有问题，都不能直接 accept。
    # ==========================================
    issues = []
    # if ???:
    #     issues.append('empty_response')
    # if ???:
    #     issues.append('duplicate_examples')
    # if ???:
    #     issues.append('padding_supervised')
    # if ???:
    #     issues.append('no_supervised_tokens')
    # if ???:
    #     issues.append('merge_not_checked')
    # if ???:
    #     issues.append('sanity_generation_not_checked')
    return {'ready': len(issues) == 0, 'issues': issues}

def recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03):
    # ==========================================
    # TODO 10: 根据项目汇总和交付检查给出采用建议
    # 规则：
    # - 数据、loss 或 artifact 未准备好：tune
    # - 参数节省达标且 val loss 损失可接受：accept
    # - 参数节省达标但 val loss 损失偏大：tune
    # - 参数节省不达标：reject
    # ==========================================
    # if ???:
    #     decision = ???
    #     reason = ???
    # elif ???:
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
def test_lora_project_template():
    try:
        examples = [
            {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
            {'prompt': '问：如何检查 loss？', 'response': '答：检查 labels 中参与监督的 token。'},
            {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
            {'prompt': '问：空回答？', 'response': ''},
        ]
        audit = audit_sft_examples(examples, max_total_chars=30)
        assert audit['total_samples'] == 4, "样本数统计不正确！"
        assert audit['empty_response_count'] == 1, "空 response 统计不正确！"
        assert audit['duplicate_count'] == 1, "重复样本统计不正确！"
        assert audit['over_length_count'] == 1, "超长样本统计不正确！"

        mask = [[1, 1, 1, 0], [1, 1, 0, 0]]
        labels = [[-100, 7, 8, -100], [-100, 9, -100, 3]]
        report = loss_mask_report(mask, labels)
        assert report['total_tokens'] == 8, "total_tokens 统计不正确！"
        assert report['non_padding_tokens'] == 5, "non_padding_tokens 统计不正确！"
        assert report['supervised_tokens'] == 4, "supervised_tokens 统计不正确！"
        assert report['padding_supervised_tokens'] == 1, "padding_supervised_tokens 统计不正确！"
        assert report['supervised_ratio'] == 0.8, "supervised_ratio 计算不正确！"

        config = build_lora_project_config(
            base_model='tiny-llama',
            target_modules=['q_proj', 'v_proj'],
            rank=8,
            alpha=16,
            dropout=0.05,
            learning_rate=2e-4,
            micro_batch_size=2,
            accum_steps=4,
            scheduler='wsd-cosine',
        )
        assert config['effective_batch_size'] == 8, "effective_batch_size 计算不正确！"
        assert config['target_modules'] == ['q_proj', 'v_proj'], "target_modules 应保留原始配置！"

        trainable = lora_trainable_params(8, 8, 2)
        total = full_linear_params(8, 8)
        ratio = lora_param_ratio(8, 8, 2)

        assert trainable == 32, "LoRA 可训练参数量计算不正确！"
        assert total == 64, "完整线性层参数量计算不正确！"
        assert abs(ratio - 0.5) < 1e-12, "LoRA 参数占比计算不正确！"

        baseline = {
            'trainable_params': 1000,
            'step_time_ms': 20.0,
            'peak_mem_mb': 1024.0,
            'final_train_loss': 0.40,
            'final_val_loss': 0.50,
        }
        lora = {
            'trainable_params': 100,
            'step_time_ms': 22.0,
            'peak_mem_mb': 768.0,
            'final_train_loss': 0.42,
            'final_val_loss': 0.52,
        }
        summary = summarize_lora_project(baseline, lora)
        assert summary['param_reduction'] == 0.9, "param_reduction 计算不正确！"
        assert summary['peak_mem_delta_mb'] == 256.0, "peak_mem_delta_mb 计算不正确！"
        assert summary['step_time_delta_ms'] == -2.0, "step_time_delta_ms 计算不正确！"
        assert summary['final_train_loss_delta'] == 0.02, "final_train_loss_delta 计算不正确！"
        assert summary['final_val_loss_delta'] == 0.02, "final_val_loss_delta 计算不正确！"

        artifact = build_adapter_artifact_record(
            adapter_path='outputs/lora-adapter',
            tokenizer_path='outputs/tokenizer',
            merge_checked=True,
            sanity_generation_checked=True,
        )
        clean_audit = {'total_samples': 2, 'empty_response_count': 0, 'duplicate_count': 0, 'over_length_count': 0, 'avg_total_chars': 12.0}
        clean_report = {'total_tokens': 8, 'non_padding_tokens': 5, 'supervised_tokens': 3, 'padding_supervised_tokens': 0, 'supervised_ratio': 0.6}
        readiness = check_lora_project_readiness(clean_audit, clean_report, artifact)
        assert readiness['ready'] is True, "干净项目应允许交付！"

        dirty_readiness = check_lora_project_readiness(audit, report, artifact)
        assert dirty_readiness['ready'] is False, "存在数据或 mask 问题时不能交付！"
        assert 'empty_response' in dirty_readiness['issues'], "应报告空 response 问题！"
        assert 'padding_supervised' in dirty_readiness['issues'], "应报告 padding 参与 loss 问题！"

        decision = recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03)
        assert decision['decision'] == 'accept', "LoRA 决策应为 accept！"

        assert recommend_lora_decision(summary, dirty_readiness)['decision'] == 'tune', "交付检查未通过时应建议 tune！"

        worse_summary = dict(summary)
        worse_summary['final_val_loss_delta'] = 0.08
        assert recommend_lora_decision(worse_summary, readiness)['decision'] == 'tune', "val loss 损失过大时应建议 tune！"

        weak_summary = dict(summary)
        weak_summary['param_reduction'] = 0.2
        assert recommend_lora_decision(weak_summary, readiness)['decision'] == 'reject', "参数节省不足时应建议 reject！"

        print("✅ LoRA 项目数据审计、loss 核对、账本、交付检查和决策代码通过基础校验。")

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


test_lora_project_template()

```

🛑 **STOP HERE** 🛑

## 参考代码与解析

### 代码


```python
# TODO 1: 审计 SFT 样本
def audit_sft_examples(examples, max_total_chars):
    seen = set()
    total_chars = 0
    empty_response_count = 0
    duplicate_count = 0
    over_length_count = 0

    for example in examples:
        prompt = example.get('prompt', '')
        response = example.get('response', '')
        pair = (prompt, response)
        total = len(prompt) + len(response)

        total_chars += total
        if not response.strip():
            empty_response_count += 1
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)
        if total > max_total_chars:
            over_length_count += 1

    total_samples = len(examples)
    avg_total_chars = total_chars / total_samples if total_samples else 0.0
    return {
        'total_samples': total_samples,
        'empty_response_count': empty_response_count,
        'duplicate_count': duplicate_count,
        'over_length_count': over_length_count,
        'avg_total_chars': round(avg_total_chars, 2),
    }

# TODO 2: 核对 loss mask
def loss_mask_report(attention_mask, labels, ignore_index=-100):
    mask_flat = [value for row in attention_mask for value in row]
    labels_flat = [value for row in labels for value in row]

    if len(mask_flat) != len(labels_flat):
        raise ValueError('attention_mask and labels must have the same number of tokens')

    total_tokens = len(labels_flat)
    non_padding_tokens = sum(1 for mask in mask_flat if mask == 1)
    supervised_tokens = sum(1 for label in labels_flat if label != ignore_index)
    padding_supervised_tokens = sum(
        1 for mask, label in zip(mask_flat, labels_flat)
        if mask == 0 and label != ignore_index
    )
    supervised_ratio = supervised_tokens / non_padding_tokens if non_padding_tokens else 0.0
    return {
        'total_tokens': total_tokens,
        'non_padding_tokens': non_padding_tokens,
        'supervised_tokens': supervised_tokens,
        'padding_supervised_tokens': padding_supervised_tokens,
        'supervised_ratio': round(supervised_ratio, 4),
    }

# TODO 3: 汇总 LoRA 项目配置
def build_lora_project_config(
    base_model,
    target_modules,
    rank,
    alpha,
    dropout,
    learning_rate,
    micro_batch_size,
    accum_steps,
    scheduler,
):
    effective_batch_size = micro_batch_size * accum_steps
    return {
        'base_model': base_model,
        'target_modules': target_modules,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'micro_batch_size': micro_batch_size,
        'accum_steps': accum_steps,
        'effective_batch_size': effective_batch_size,
        'scheduler': scheduler,
    }

# TODO 4: 计算单层 LoRA 的可训练参数量
def lora_trainable_params(in_dim, out_dim, rank):
    """Estimate trainable LoRA parameters for a single linear layer."""
    trainable_params = rank * (in_dim + out_dim)
    return trainable_params

# TODO 5: 计算完整线性层的参数量
def full_linear_params(in_dim, out_dim):
    total_params = in_dim * out_dim
    return total_params

# TODO 6: 计算 LoRA 参数占比
def lora_param_ratio(in_dim, out_dim, rank):
    trainable = lora_trainable_params(in_dim, out_dim, rank)
    total = full_linear_params(in_dim, out_dim)
    ratio = trainable / total
    return ratio

# TODO 7: 汇总 baseline 和 LoRA 项目指标
def summarize_lora_project(baseline_metrics, lora_metrics):
    param_reduction = 1.0 - lora_metrics['trainable_params'] / baseline_metrics['trainable_params']
    memory_delta = baseline_metrics['peak_mem_mb'] - lora_metrics['peak_mem_mb']
    time_delta = baseline_metrics['step_time_ms'] - lora_metrics['step_time_ms']
    train_loss_delta = lora_metrics['final_train_loss'] - baseline_metrics['final_train_loss']
    val_loss_delta = lora_metrics['final_val_loss'] - baseline_metrics['final_val_loss']
    return {
        'param_reduction': round(param_reduction, 4),
        'peak_mem_delta_mb': round(memory_delta, 2),
        'step_time_delta_ms': round(time_delta, 2),
        'final_train_loss_delta': round(train_loss_delta, 4),
        'final_val_loss_delta': round(val_loss_delta, 4),
    }

# TODO 8: 记录 adapter 交付物
def build_adapter_artifact_record(adapter_path, tokenizer_path, merge_checked, sanity_generation_checked):
    return {
        'adapter_path': adapter_path,
        'tokenizer_path': tokenizer_path,
        'merge_checked': merge_checked,
        'sanity_generation_checked': sanity_generation_checked,
    }

# TODO 9: 检查项目是否可以交付
def check_lora_project_readiness(data_audit, mask_report, artifact_record):
    issues = []
    if data_audit['empty_response_count'] > 0:
        issues.append('empty_response')
    if data_audit['duplicate_count'] > 0:
        issues.append('duplicate_examples')
    if mask_report['padding_supervised_tokens'] > 0:
        issues.append('padding_supervised')
    if mask_report['supervised_tokens'] == 0:
        issues.append('no_supervised_tokens')
    if not artifact_record['merge_checked']:
        issues.append('merge_not_checked')
    if not artifact_record['sanity_generation_checked']:
        issues.append('sanity_generation_not_checked')
    return {'ready': len(issues) == 0, 'issues': issues}

# TODO 10: 根据项目汇总和交付检查给出采用建议
def recommend_lora_decision(summary, readiness, min_param_reduction=0.5, max_val_loss_delta=0.03):
    if not readiness['ready']:
        decision = 'tune'
        reason = '数据、loss mask 或 adapter 交付检查未通过，先修复项目可信度问题。'
    elif summary['param_reduction'] < min_param_reduction:
        decision = 'reject'
        reason = '参数节省不足，LoRA 没有带来足够训练成本收益。'
    elif summary['final_val_loss_delta'] > max_val_loss_delta:
        decision = 'tune'
        reason = '参数节省达标，但验证集 loss 损失偏大，优先调 rank、target modules 或学习率。'
    else:
        decision = 'accept'
        reason = '参数节省达标，验证集损失可接受，交付检查通过，可以保留当前 LoRA 配置。'
    return {'decision': decision, 'reason': reason}

examples = [
    {'prompt': '问：什么是 LoRA？', 'response': '答：LoRA 是低秩适配方法。'},
    {'prompt': '问：如何检查 loss？', 'response': '答：检查 labels 中参与监督的 token。'},
]
audit = audit_sft_examples(examples, max_total_chars=64)
print(audit)

mask_report = loss_mask_report(
    attention_mask=[[1, 1, 1, 0]],
    labels=[[-100, 7, 8, -100]],
)
print(mask_report)

config = build_lora_project_config(
    base_model='tiny-llama',
    target_modules=['q_proj', 'v_proj'],
    rank=8,
    alpha=16,
    dropout=0.05,
    learning_rate=2e-4,
    micro_batch_size=2,
    accum_steps=4,
    scheduler='wsd-cosine',
)
print(config)

for hidden_size, rank in [(4096, 8), (4096, 16), (8192, 16)]:
    trainable = lora_trainable_params(hidden_size, hidden_size, rank)
    total = full_linear_params(hidden_size, hidden_size)
    ratio = lora_param_ratio(hidden_size, hidden_size, rank)
    print(f"hidden={hidden_size}, rank={rank} -> trainable={trainable:,}, full={total:,}, ratio={ratio:.4%}")

baseline = {'trainable_params': 1000, 'step_time_ms': 20.0, 'peak_mem_mb': 1024.0, 'final_train_loss': 0.40, 'final_val_loss': 0.50}
lora = {'trainable_params': 100, 'step_time_ms': 22.0, 'peak_mem_mb': 768.0, 'final_train_loss': 0.42, 'final_val_loss': 0.52}
summary = summarize_lora_project(baseline, lora)
artifact = build_adapter_artifact_record('outputs/lora-adapter', 'outputs/tokenizer', True, True)
readiness = check_lora_project_readiness(audit, mask_report, artifact)
print(summary)
print(readiness)
print(recommend_lora_decision(summary, readiness))

```

### 解析

**1. TODO 1: 审计 SFT 样本**
- **实现方式**：遍历 `prompt / response` 样本，统计总样本数、空 response、重复样本、超长样本和平均长度。
- **关键点**：微调前先确认数据可信。空 response 会让样本没有有效监督，重复样本会放大小数据过拟合风险，超长样本会改变截断和显存口径。
- **项目意义**：这一步把第 09 节的数据正确性从单条样本扩展到项目级数据集检查。

**2. TODO 2: 核对 loss mask**
- **实现方式**：把 `attention_mask` 和 `labels` 展平后对齐检查，统计非 padding token、参与监督的 token，以及 padding 中错误参与 loss 的 token。
- **关键点**：`labels != -100` 的 token 会参与 loss；`attention_mask == 0` 的 padding token 不应该参与 loss。
- **项目意义**：这是 SFT 项目最关键的正确性检查之一。loss 下降不代表训练对了，必须确认监督 token 的位置正确。

**3. TODO 3: 汇总 LoRA 项目配置**
- **实现方式**：把 base model、target modules、rank、alpha、dropout、学习率、micro batch、accum steps 和 scheduler 放进同一个配置对象。
- **关键点**：`effective_batch_size = micro_batch_size * accum_steps`，这要和第 12 节的梯度累积口径一致。
- **项目意义**：LoRA 项目必须能复现；只报告 loss 和显存，不记录配置，后续无法判断差异来自哪里。

**4. TODO 4: 计算单层 LoRA 的可训练参数量**
- **实现方式**：LoRA 为一个线性层增加两个低秩矩阵，`A` 的参数量是 `rank * in_dim`，`B` 的参数量是 `rank * out_dim`，合起来是 `rank * (in_dim + out_dim)`。
- **关键点**：这里统计的是 LoRA adapter 的可训练参数，不包括冻结的底座权重。
- **项目意义**：这是 LoRA 微调项目的第一张账本，用来说明训练侧到底少更新了多少参数。

**5. TODO 5: 计算完整线性层的参数量**
- **实现方式**：完整线性层的 weight 参数量是 `in_dim * out_dim`。本节为了突出主线，不额外统计 bias。
- **关键点**：全参线性层是 baseline，用来衡量 LoRA 的参数节省比例。
- **技术细节**：如果真实模型中包含 bias 或多个投影层，需要把这些层逐项累加。

**6. TODO 6: 计算 LoRA 参数占比**
- **实现方式**：先分别计算 LoRA 参数量和完整线性层参数量，再用 `trainable / total` 得到参数占比。
- **关键点**：参数占比越小，说明同一层上需要训练和保存的 adapter 越少。
- **项目意义**：这个比例可以和 step time、peak memory、train/val loss 一起放进项目报告，不能单独作为最终结论。

**7. TODO 7: 汇总 baseline 和 LoRA 项目指标**
- **实现方式**：资源类指标使用 `baseline - LoRA`，正数表示 LoRA 更省或更快；loss 指标使用 `LoRA - baseline`，正数表示 LoRA 效果更差。
- **关键点**：train loss 和 val loss 要分开看。train loss 接近不代表泛化可接受，最终决策更应该看 val loss delta。
- **工程判断**：如果参数和显存明显下降，但 val loss 损失很小，LoRA 方案通常值得保留；如果 val loss 明显变差，需要继续调整 rank、插层位置或学习率。

**8. TODO 8: 记录 adapter 交付物**
- **实现方式**：记录 adapter 路径、tokenizer 路径、merge 检查和最小生成样例检查。
- **关键点**：LoRA 微调的交付物不是一行 loss，而是一组可加载、可复现、能做 sanity check 的 artifact。
- **项目意义**：这一步把训练实验推进到交付边界，避免“训练完但无法复现或无法加载”。

**9. TODO 9: 检查项目是否可以交付**
- **实现方式**：把数据审计、loss mask 报告和 artifact 记录合并检查，返回 `ready` 和问题列表。
- **关键点**：只要存在空 response、padding 参与 loss、无监督 token、merge 未检查或生成样例未检查，就不应该直接把项目判为 accept。
- **项目意义**：这一步让项目报告不只比较指标，也能说明指标是否可信。

**10. TODO 10: 输出采用建议**
- **accept**：交付检查通过，参数节省达标，val loss 损失在阈值内。
- **tune**：交付检查未通过，或参数节省达标但 val loss 损失偏大。
- **reject**：交付检查通过，但参数节省不足，LoRA 没有带来足够训练成本收益。
- **项目意义**：决策不再只看 LoRA 参数比例，而是同时看数据可信度、loss 口径、artifact 交付、资源收益和效果损失。
