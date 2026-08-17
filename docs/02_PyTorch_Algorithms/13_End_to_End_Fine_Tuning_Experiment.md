# 13. End to End Fine Tuning Experiment | 端到端微调实验

**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `SFT`, `训练闭环` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

前面的小节已经分别讲过模型封装、优化器、损失函数和梯度累积，但真实微调不是把这些概念单独跑通就结束。只要数据构造、label 对齐、loss 计算或参数更新里有一个环节接错，训练就会表现成 loss 不降、shape 对不上，或者看似运行但模型没有真正学习。

本节把这些训练要素收成一个最小端到端 SFT 实验：先构造 train / val 样本，再计算自回归 loss，最后走完 backward、梯度累积、optimizer step 和周期性评估。完成后，你应该能用一个小模型验证完整微调闭环是否跑通，也能输出一份最小训练报告，为后面的 LoRA 项目、RLHF/PPO 和训练性能分析建立统一基线。

**关键词：** `end-to-end`, `fine-tuning`, `train/val`, `report`

---
## 前置阅读

**导语：** 先把 SFT 数据与 loss 对齐、LoRA 适配、优化器、梯度累积、学习率调度和最小训练接口看过，再做端到端微调实验最顺。
- [09. SFT Training Loop | 监督微调训练循环](../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [10. LoRA Tutorial | LoRA 教程](../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [12. Gradient Accumulation | 梯度累积](../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- [11. LR Schedulers WSD Cosine | WSD 余弦学习率调度器](../02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.md)
- [P0: 12. PyTorch Minimal Training Interface | PyTorch 最小训练接口](../00_Prerequisites/12_PyTorch_Minimal_Training_Interface.md)

## 相关阅读

**导语：** 完成最小 SFT 闭环后，下一步最自然的是把它推进到 LoRA 项目、指令微调项目和训练性能分析里。
- [60. LoRA Fine-Tuning Project | LoRA 微调项目](../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)
- [62. Instruction Fine-Tuning Project | 指令微调项目](../02_PyTorch_Algorithms/62_Instruction_Fine_Tuning_Project.md)
- [73. Training Performance Analysis | 训练性能分析](../02_PyTorch_Algorithms/73_Training_Performance_Analysis.md)
- [74. Profiling-Driven End-to-End Optimization | Profiling 驱动的端到端优化](../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.md)

---
### Step 1: 端到端训练闭环长什么样
端到端微调实验的核心，是把数据、模型、loss、优化器和评估五层接成一个可运行闭环。

一个完整的微调实验通常包含五层：
1. **数据层**：将 prompt/response 构造为 tokenized batch（input_ids + attention_mask + labels），并进行 padding 对齐，作为模型的直接输入。
2. **模型层**：输入 token 经过 embedding -> Transformer / RNN -> LM head，输出每个位置的 logits。
3. **优化层**：计算 SFT loss，执行 backward、step 和 zero_grad。
4. **训练控制层**：控制梯度累积、参数更新频率和 loss 记录。
5. **评估层**：在训练中定期记录 train / val loss，并用小样本 overfit 检查确认闭环真的接通。

这一页承担 `00-12` 的阶段性项目收口：用一个极小语言模型，把前面的训练组件串成完整闭环。后面的 `TODO 1-4` 会分别把数据、loss、评估和训练报告拆开实现，再重新合回一个训练闭环。

### Step 2: 为什么要把它做成实验
先说明为什么要把单点函数串成完整实验，再进入代码。

如果只会单点函数，很容易在真实项目里出现“会公式，不会落地”的问题。端到端实验的价值在于从确认接口正确，到观察训练收敛，再到快速定位问题，最后用极端测试验证闭环：
- 你能确认数据、模型、loss、优化器之间的接口是对的。例如：模型的输出 shape 是否匹配 loss 函数的输入 shape？优化器的参数是否真的被更新？
- 你能观察训练 loss 是否真的下降，并判断验证 loss 是否同步变化。
- 你能快速定位是数据问题、loss 问题、优化器问题，还是评估口径问题。
- 你能通过“重复样本过拟合测试”快速验证闭环是否跑通：如果模型在重复样本上 loss 能显著下降，说明数据、loss、优化器链路完整；如果 loss 不降，说明链路中有环节断裂。

注意：这里的 val batch 仍然是 sanity check，不代表真实泛化能力。真实项目里还需要更大的验证集、任务指标和样例回归测试。

### Step 3: 代码实现框架与任务拆解

本实验的模型层（TinyCausalLM）已直接给出，无需修改。它是一个极小的自回归模型（embedding -> GRU -> LM head），参数规模极小，便于快速验证闭环。

四个 TODO 与训练闭环的对应关系如下：
- TODO 1 -> 数据构造（build_sft_batch / collate_sft_batch）
- TODO 2 -> 损失计算（compute_sft_loss）
- TODO 3 -> 评估函数（evaluate_loss）
- TODO 4 -> 训练更新与报告（run_finetuning_experiment：backward、梯度累积、optimizer step、train/val history）
- 模型层 -> TinyCausalLM 已给出，用于验证闭环，不作为 TODO

下面会实现四块代码：
- `build_sft_batch`：将一条 prompt/response 转为 `input_ids / attention_mask / labels`。
- `collate_sft_batch`：把多条样本堆成 batch。
- `compute_sft_loss`：完成 next-token 对齐并计算 SFT loss。
- `run_finetuning_experiment`：驱动训练循环，返回包含初始 loss、最终 loss 和历史记录的报告。

#### 实现顺序

建议先实现数据构造，再实现 loss，然后实现评估函数，最后把它们串成训练闭环：

1. `build_sft_batch` / `collate_sft_batch`：先把样本变成能喂给模型的三件套。
2. `compute_sft_loss`：再把 logits 和 labels 对齐，确认监督口径没问题。
3. `evaluate_loss`：用同一套口径检查 train / val loss。
4. `run_finetuning_experiment`：最后把数据、loss、optimizer 和 report 串成完整闭环。

#### 实现节奏

这页不是把四个 TODO 一次性堆出来，而是按“先数据、再损失、再评估、最后训练报告”的顺序推进。这样写的原因很简单：

- 如果 `build_sft_batch` 有问题，后面的 loss 和训练报告都不可信。
- 如果 `compute_sft_loss` 的对齐不对，模型可能在错误位置上学习。
- 如果 `evaluate_loss` 和训练口径不一致，报告没有解释力。
- 只有把前面三层都对齐后，`run_finetuning_experiment` 才真正有意义。
#### 图解：端到端微调闭环

`13` 不是再多写一个 loss 函数，而是把前面几页接成一条能验证的实验链路。

```text
samples
  │
  ▼
collate_sft_batch
  │  input_ids / attention_mask / labels
  ▼
TinyCausalLM ─────► logits
  │                 │
  │                 ▼
  │          compute_sft_loss
  │                 │
  ▼                 ▼
train micro-batches + gradient accumulation
  │
  ▼
optimizer.step()
  │
  ├──► evaluate train loss
  ├──► evaluate val loss
  └──► report: initial/final/history
```

最小报告至少回答三件事：
- 初始 train / val loss 是多少。
- 训练后 train / val loss 是否下降。
- 重复样本 sanity check 是否能快速 overfit。

![端到端训练闭环](/02_PyTorch_Algorithms/13_training_loop.svg)

#### 最小报告模板与判据

端到端实验跑完以后，真正需要留下来的不是一堆日志，而是一份能回答“值不值得继续做”的最小报告。

| 报告字段 | 含义 | 你要怎么看 |
|:---|:---|:---|
| `initial_train_loss` | 训练前的基线 loss | 用来确认实验有没有起点 |
| `initial_val_loss` | 验证前的基线 loss | 用来检查 train / val 是否一致起跑 |
| `history` | 中间训练过程记录 | 看 loss 是否稳定下降、是否有波动 |
| `final_train_loss` | 训练后的最终 loss | 看模型是否真的学到了样本模式 |
| `final_val_loss` | 验证后的最终 loss | 看训练结果是否只是在记忆训练集 |

最小判据建议分成三层：

1. **闭环判据**：`train / val / optimizer` 这三条链路都能跑通，且评估函数与训练 loss 同口径。
2. **学习判据**：重复样本或极小数据上，loss 应该能明显下降，说明梯度、数据和更新路径接通。
3. **解释判据**：如果 train loss 降了但 val loss 不变，要优先怀疑数据分布、样本量和评估口径，而不是急着改模型结构。

真实项目里，报告通常还要补样例输出、错误案例和任务指标；但在这节里，先把“能训练、能评估、能解释”这三件事做实就够了。

```python
import torch
import torch.nn as nn

```


```python
def build_sft_batch(prompt_ids, response_ids, pad_id=0, eos_id=2, max_len=10):
    # ==========================================
    # TODO 1: 构造单条 SFT 样本
    # 提示: prompt 部分的 labels mask 为 -100，response/EOS 部分保留
    #       返回 input_ids、attention_mask、labels 三个 tensor
    # ==========================================
    response_with_eos = response_ids + [eos_id]
    # input_ids = ???
    # labels = ???

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    if not any(label != -100 for label in labels):
        raise ValueError("截断后没有有效监督 token")

    # attention_mask = ???
    # pad_len = ???
    # input_ids = ???
    # attention_mask = ???
    # labels = ???

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_sft_batch(samples, pad_id=0, eos_id=2, max_len=10):
    items = [build_sft_batch(prompt, response, pad_id=pad_id, eos_id=eos_id, max_len=max_len) for prompt, response in samples]
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels, attention_mask=None):
    # ==========================================
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    # 提示: logits 取前 t-1 个位置，labels 取后 t-1 个位置
    #       如果传入 attention_mask，也同步保护 padding 位置
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    # if attention_mask is not None:
    #     shift_attention_mask = ???
    #     shift_labels = ???
    # if ???:
    #     raise ValueError(...)
    # loss = ???
    return loss


def evaluate_loss(model, batch):
    # ==========================================
    # TODO 3: 在 eval 模式下计算 batch loss
    # ==========================================
    # model.eval()
    # with torch.no_grad():
    #     logits = ???
    #     loss = ???
    # return ???
    pass


def run_finetuning_experiment(model, optimizer, train_batch, val_batch=None, accum_steps=2, num_updates=40, eval_every=10):
    """
    在小批样本上反复训练，观察端到端训练闭环是否跑通，并返回最小训练报告。
    """
    if train_batch["input_ids"].size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    # ==========================================
    # TODO 4: 端到端训练闭环与报告
    # 提示: 记录初始 train/val loss -> micro-batch 累积梯度 -> 定期评估 -> 返回 report
    # ==========================================
    # report = ???
    # micro_size = ???
    # for step in range(...):
    #     model.train()
    #     optimizer.zero_grad()
    #     for idx in range(accum_steps):
    #         mb = ???
    #         logits = ???
    #         loss = ???
    #         loss.backward()
    #     optimizer.step()
    #     if ???:
    #         report["history"].append(...)
    # report["final_train_loss"] = ???
    # report["final_val_loss"] = ???
    # return report
    pass

```


```python
# 运行此单元格以测试你的实现
def test_end_to_end_finetuning():
    try:
        torch.manual_seed(7)

        train_samples = [
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
        ]
        val_samples = [
            ([1, 2, 3], [4, 5, 6, 7]),
            ([1, 2, 3], [4, 5, 6, 7]),
        ]

        train_batch = collate_sft_batch(train_samples, pad_id=0, eos_id=2, max_len=9)
        val_batch = collate_sft_batch(val_samples, pad_id=0, eos_id=2, max_len=9)

        assert train_batch["input_ids"].shape == (4, 9), "train batch shape 错误"
        assert train_batch["attention_mask"].sum().item() == 32, "attention_mask 统计错误"
        assert torch.any(train_batch["labels"] == -100), "prompt/padding 应该被 mask"

        model = TinyCausalLM(vocab_size=64, hidden_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

        report = run_finetuning_experiment(
            model,
            optimizer,
            train_batch,
            val_batch=val_batch,
            accum_steps=2,
            num_updates=30,
            eval_every=10,
        )

        print(f"Initial train loss: {report['initial_train_loss']:.4f}")
        print(f"Final train loss  : {report['final_train_loss']:.4f}")
        print(f"Initial val loss  : {report['initial_val_loss']:.4f}")
        print(f"Final val loss    : {report['final_val_loss']:.4f}")
        print(f"History           : {report['history']}")

        assert len(report["history"]) >= 3, "训练过程应该至少记录 3 次评估"
        assert report["final_train_loss"] < report["initial_train_loss"], "训练没有让 train loss 下降"
        assert report["final_val_loss"] < report["initial_val_loss"], "训练没有让 val loss 下降"
        assert report["final_train_loss"] < 0.2, "重复样本过拟合不充分，闭环可能有问题"

        print("✅ 测试通过！端到端微调闭环、评估和报告均运行正常。")
    except NotImplementedError:
        print("请先完成 TODO 部分。")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分。") from e
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

test_end_to_end_finetuning()

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
import torch
import torch.nn as nn


def build_sft_batch(prompt_ids, response_ids, pad_id=0, eos_id=2, max_len=10):
    # TODO 1: 构造单条 SFT 样本
    response_with_eos = response_ids + [eos_id]
    input_ids = prompt_ids + response_with_eos
    labels = [-100] * len(prompt_ids) + response_with_eos

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    if not any(label != -100 for label in labels):
        raise ValueError("截断后没有有效监督 token")

    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_sft_batch(samples, pad_id=0, eos_id=2, max_len=10):
    items = [build_sft_batch(prompt, response, pad_id=pad_id, eos_id=eos_id, max_len=max_len) for prompt, response in samples]
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        hidden, _ = self.rnn(x)
        logits = self.lm_head(hidden)
        return logits


def compute_sft_loss(logits, labels, attention_mask=None):
    # TODO 2: 对齐 next-token 预测并计算 SFT loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)
    if not torch.any(shift_labels != -100):
        raise ValueError("当前 batch 没有有效监督 token")

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def evaluate_loss(model, batch):
    # TODO 3: 在 eval 模式下计算 batch loss
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch.get("attention_mask"))
        loss = compute_sft_loss(logits, batch["labels"], batch.get("attention_mask"))
    return loss.item()


def run_finetuning_experiment(model, optimizer, train_batch, val_batch=None, accum_steps=2, num_updates=40, eval_every=10):
    # TODO 4: 端到端训练闭环与报告
    if train_batch["input_ids"].size(0) % accum_steps != 0:
        raise ValueError("batch size 必须能被 accum_steps 整除")

    report = {
        "initial_train_loss": evaluate_loss(model, train_batch),
        "initial_val_loss": evaluate_loss(model, val_batch) if val_batch is not None else None,
        "final_train_loss": None,
        "final_val_loss": None,
        "history": [],
    }
    micro_size = train_batch["input_ids"].size(0) // accum_steps

    for step in range(1, num_updates + 1):
        model.train()
        optimizer.zero_grad()

        for idx in range(accum_steps):
            start = idx * micro_size
            end = (idx + 1) * micro_size
            mb = {key: value[start:end] for key, value in train_batch.items()}
            logits = model(mb["input_ids"], mb.get("attention_mask"))
            loss = compute_sft_loss(logits, mb["labels"], mb.get("attention_mask")) / accum_steps
            loss.backward()

        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == num_updates:
            record = {"step": step, "train_loss": evaluate_loss(model, train_batch)}
            if val_batch is not None:
                record["val_loss"] = evaluate_loss(model, val_batch)
            report["history"].append(record)

    report["final_train_loss"] = evaluate_loss(model, train_batch)
    report["final_val_loss"] = evaluate_loss(model, val_batch) if val_batch is not None else None
    return report

```

### 答案与直觉

- **这一题要解决什么**：把 SFT 数据构造、loss 计算、训练更新和评估报告串成一个最小闭环。
- **为什么这样做**：只有把输入、监督信号、优化器路径和评估口径全部对齐，实验结果才有可解释性。
- **带走的直觉**：端到端实验的重点不是堆功能，而是确认整条训练链路能稳定跑通，并能用报告判断它是否真的学习。

**1. TODO 1 (构造 SFT batch)**

- **拼接输入**：`input_ids` 由 `prompt + response + EOS` 拼接得到，保持样本的完整上下文和结束信号。
- **监督标签**：`labels` 里，`prompt` 对应的位置要 mask 成 `-100`，只让模型学习 response 和 EOS。
- **attention mask**：真实 token 为 `1`，padding 为 `0`，和 `labels=-100` 分工不同。
- **长度处理**：超过 `max_len` 时要裁剪，不足时要补 `pad_id`、`attention_mask=0` 和 `labels=-100`。

**2. TODO 2 (对齐 next-token 并计算 SFT loss)**

- **一位错位**：`shift_logits = logits[..., :-1, :]`，`shift_labels = labels[..., 1:]`。
- **损失函数**：使用 `CrossEntropyLoss(ignore_index=-100)` 计算 loss，让 prompt 和 padding 位置自然忽略。
- **监督范围**：训练信号只来自 response / EOS 的有效 token，next-token 对齐要和 causal LM 的训练目标一致。
- **防御检查**：如果 batch 里没有任何有效 label，要直接报错，否则 loss 没有训练意义。

**3. TODO 3 (评估函数)**

- **eval 模式**：评估时调用 `model.eval()` 并用 `torch.no_grad()` 关闭梯度记录。
- **同口径 loss**：训练和验证都调用同一个 `compute_sft_loss`，避免评估口径和训练口径不一致。
- **工程价值**：真实项目里 loss 之外还需要任务指标和样例回归，但最小闭环先保证 loss 口径正确。

**4. TODO 4 (训练闭环与报告)**

- **micro-batch**：先把 batch 切成多个 `micro-batch`，再逐个累积梯度。
- **loss 缩放**：每个 `micro-batch` 的 loss 要除以 `accum_steps`，保证和完整 batch 的梯度一致。
- **周期评估**：在第 1 步、固定间隔和最后一步记录 train / val loss，形成最小训练报告。
- **参数更新**：所有 `micro-batch` 处理完之后，再统一执行 `optimizer.step()`。

**进阶思考：为什么要做重复样本验证？**

- **一致性检查**：通过重复样本验证，可以确认数据构造、loss 对齐、梯度累积和参数更新是否真的接通。
- **闭环意义**：如果重复样本都不能快速 overfit，真实数据上的微调结果通常也不可信。
- **工程边界**：这不是泛化评估，只是 sanity check；真实微调还要补验证集指标、样例回归和错误案例分析。
