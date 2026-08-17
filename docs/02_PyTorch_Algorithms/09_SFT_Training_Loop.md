# 09. SFT Training Loop | 监督微调训练循环

**难度：** Medium | **环境：** CPU-first | **标签：** `训练微调`, `SFT`, `训练循环` | **目标人群：** 训练机制学习者

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/09_SFT_Training_Loop.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


---

## 本节导读

把模型结构写出来以后，下一步就是让它按监督数据学习回答。但 SFT 最容易出错的地方并不在 optimizer，而在数据和 loss 的对齐：模型输入通常是 `[prompt + response]`，真正应该学习的是 response，而不是让模型去复述 prompt。

本节聚焦 SFT 训练循环里最关键的三件事：用 prompt masking 把不该学习的位置设为 `ignore_index`，用 `attention_mask` 区分真实 token 和 padding，再通过 shift logits / labels 对齐下一个 token 预测。完成后，你应该能看懂 `input_ids`、`attention_mask`、`labels` 和 cross entropy 之间的关系，并为后面的端到端微调实验、LoRA 和 RLHF 对齐训练打基础。

**关键词：** `SFT`, `masking`, `attention_mask`, `shift logits`

---
## 前置阅读

**导语：** 先把模型封装、训练循环和优化器基础看清，再读 SFT 的数据构造与 loss 对齐会更顺。

- [P0: 09. PyTorch nn.Module Basics | PyTorch nn.Module 基础](../00_Prerequisites/09_PyTorch_nn_Module_Basics.md)
- [P0: 11. PyTorch Optimizers and Loss | PyTorch 优化器与损失](../00_Prerequisites/11_PyTorch_Optimizers_and_Loss.md)
- [P0: 13. Simple Neural Network Training | 简单神经网络训练循环](../00_Prerequisites/13_Simple_Neural_Network_Training.md)

## 相关阅读

**导语：** 读完 SFT 的最小训练循环后，建议继续看端到端微调实验、LoRA 以及训练显存和性能分析。

- [10. LoRA Tutorial | LoRA 教程](../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- [13. End-to-End Fine-Tuning Experiment | 端到端微调实验](../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)
- [P0: 17. PyTorch Profiling Basics | PyTorch 性能剖析基础](../00_Prerequisites/17_PyTorch_Profiling_Basics.md)
- [P0: 18. Memory Profiling and Optimization | 显存剖析与优化](../00_Prerequisites/18_Memory_Profiling_and_Optimization.md)

---
### Step 1: 核心思想与痛点

SFT 和预训练的关键差异在于 loss 只应该作用在 response 上，而不是 prompt 上。

> **预训练 (Pre-training) vs 微调 (SFT)**
> * **预训练**：模型预测下一个 Token。给定一本书，每一个字都要算 Loss。
> * **SFT**：给定 `[Prompt] + [Response]`。我们**只关心**模型能不能输出正确的 `Response`。如果把 `Prompt` 也纳入 Loss 计算，模型就会去“背诵”人类的提问方式，而不是去“回答”问题。
> 
> **如何解决？（Loss Masking）**
> 在 PyTorch 的 `CrossEntropyLoss` 中，`ignore_index=-100` 的位置不会产生梯度。我们把 `labels` 中属于 Prompt 和 Padding 的位置设为 `-100`，只保留 Response 和必要的 EOS 作为监督信号。

真实微调里通常还会多一层 chat template：先把多轮 `messages` 渲染成模型约定的 prompt/response 文本，再 tokenizer 成 token id。无论模板长什么样，最后进入训练循环时都要落成三件套：

- `input_ids`：prompt、response、可选 EOS 和 padding 后的完整 token 序列。
- `attention_mask`：真实 token 为 `1`，padding 为 `0`，告诉模型哪些位置是有效上下文。
- `labels`：prompt 和 padding 为 `-100`，response / EOS 保留原 token id，告诉 loss 哪些位置要学习。

后面的 `Step 2 / Step 3` 就围绕这条链路，把 logits 对齐、loss 计算和训练流程串起来。

### Step 2: Attention Mask、Loss Mask 与 Shift Logits

`attention_mask` 和 `labels=-100` 解决的是两个不同问题，不能混在一起理解。

- `attention_mask=0`：告诉模型 padding 位置不是有效上下文，真实 Transformer 会用它避免 attention 看到 padding。
- `labels=-100`：告诉 loss 这些位置不参与监督，通常包括 prompt 和 padding。
- `EOS`：如果模板要求模型学会在回答结束时停止，EOS 应该作为 response 的一部分参与监督。

在自回归语言模型中，position `t` 的 logits 预测 position `t+1` 的 label。因此计算 CrossEntropyLoss 前要做一位 shift：`logits[..., :-1, :]` 对齐 `labels[..., 1:]`。如果截断后 response 被全部截掉，labels 里就没有有效监督 token，这种样本应该被过滤或报错，而不是静默参与训练。

#### 对齐表：input_ids / attention_mask / labels

下面这张表把一条 `[prompt + response + EOS + padding]` 样本拆开看。读 SFT 代码时，先确认这三行是否对齐，再看 loss。

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 语义 | prompt | prompt | prompt | response | response | response | response | EOS | padding |
| `input_ids` | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 2 | 0 |
| `attention_mask` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| `labels` | -100 | -100 | -100 | 40 | 50 | 60 | 70 | 2 | -100 |
| shift 后谁被预测 | 20 | 30 | 40 | 50 | 60 | 70 | 2 | 0 | - |

关键判断：
- prompt 位置可以被模型看见，但不参与 loss。
- response 和 EOS 参与 loss，模型才会学习回答和结束。
- padding 既不应该被 attention 当成有效上下文，也不应该参与 loss。

![SFT 对齐图](/02_PyTorch_Algorithms/09_sft_alignment.svg)

#### 数据审计与样本边界

在真实 SFT 数据里，`labels` 正确与否只是第一步，还要再过一层样本审计，避免“能跑但学不到”的脏样本进入训练：

| 检查项 | 你要确认什么 | 常见问题 |
|:---|:---|:---|
| `prompt` 是否可见 | prompt 只负责提供上下文，不参与 loss | prompt 被误当成监督目标 |
| `response` 是否存在 | 至少有一段可学习的回答 | 空 response、模板残缺 |
| `EOS` 是否保留 | 让模型学会结束回答 | 只学会续写，不学会停 |
| 截断后是否仍有监督 token | `max_len` 不能把 response 全截没 | 截断后 loss 全是 `-100` |
| padding 是否只在尾部 | padding 只能补在真实 token 之后 | 中间 padding 破坏 causal 结构 |

工程上最常见的坏例子有三类：

- **格式坏样本**：chat template 没渲染完整，`prompt / response` 边界不清楚。
- **监督坏样本**：`labels` 不是 `-100` 就是错位 token，loss 看似下降但学不到回答。
- **长度坏样本**：`max_len` 太短把 response 截没了，样本等于无监督数据。

所以这节的真实目标不是“把 token 拼起来”，而是先确认：**哪些 token 是上下文，哪些 token 是监督，哪些样本应该直接过滤。**
### Step 3: 动手实战

**要求**：请补全下方 `build_sft_data`（构造单条 SFT 数据）和 `compute_sft_loss`（计算损失）的 `TODO` 逻辑。

接下来把“数据构造 -> attention mask -> 标签 mask -> next-token 对齐”串成一个最小训练闭环。这里仍然使用 token id 直接演示；真实工程中的 chat template 和 tokenizer 会在进入本函数前完成。


```python
import torch
import torch.nn as nn
```


```python
def build_sft_data(
    prompt_ids: list[int],
    response_ids: list[int],
    pad_id: int = 0,
    eos_id: int | None = None,
    max_len: int = 16,
    min_response_tokens: int = 1,
):
    """
    构造单条 SFT 训练数据，返回 input_ids / attention_mask / labels。
    """
    response_with_eos = response_ids + ([] if eos_id is None else [eos_id])

    # 1. 拼接成完整序列。
    input_ids = prompt_ids + response_with_eos

    # ==========================================
    # Prompt 部分先统一标成 ignore_index，确保只对 Response/EOS 计算损失。
    # TODO 1: 构造 labels
    # 规则：
    # - 长度与 input_ids 相同
    # - prompt 部分的 label 设置为 -100
    # - response/EOS 部分的 label 保持原样
    # ==========================================
    # labels = ???

    # ==========================================
    # TODO 2: 截断 (Truncation) 与有效监督检查
    # 规则：
    # - 如果超出 max_len，从末尾截断
    # - 截断后至少保留 min_response_tokens 个可监督 token
    # ==========================================
    # input_ids = ???
    # labels = ???
    # valid_supervised = ???
    # if valid_supervised < min_response_tokens:
    #     raise ValueError(...)

    # ==========================================
    # TODO 3: attention mask 与填充 (Padding)
    # 规则：
    # - padding 前的真实 token 位置为 1
    # - input_ids 填 pad_id，attention_mask 填 0，labels 填 -100
    # ==========================================
    # attention_mask = ???
    # pad_len = ???
    # input_ids = ???
    # attention_mask = ???
    # labels = ???

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None = None):
    """
    计算自回归 SFT Loss
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]，可选，用于二次保护 padding 位置
    """
    # ==========================================
    # TODO 4: 实现 Shift 错位对齐
    # 将 logits 的最后一个 token 切掉
    # 将 labels 的第一个 token 切掉
    # 如果传入 attention_mask，也同步切掉第一个位置
    # ==========================================
    # shift_logits = ???
    # shift_labels = ???
    # if attention_mask is not None:
    #     shift_attention_mask = ???
    #     shift_labels = ???

    # ==========================================
    # TODO 5: 检查是否存在有效监督 token，并计算交叉熵
    # ==========================================
    # if ???:
    #     raise ValueError(...)
    # loss_fct = ???
    # loss = ???

    return loss

```


```python
# 运行此单元格以测试你的实现
def test_sft_pipeline():
    try:
        # --- 测试数据构造 ---
        prompt = [10, 20, 30]
        response = [40, 50, 60, 70]
        pad_id = 0
        eos_id = 2
        max_len = 9

        input_ids, attention_mask, labels = build_sft_data(prompt, response, pad_id, eos_id, max_len)

        print(f"Input IDs      : {input_ids.tolist()}")
        print(f"Attention Mask : {attention_mask.tolist()}")
        print(f"Labels         : {labels.tolist()}")

        assert input_ids.tolist() == [10, 20, 30, 40, 50, 60, 70, 2, 0], "Input IDs 构造错误！"
        assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0], "attention_mask 构造错误！"
        assert labels.tolist() == [-100, -100, -100, 40, 50, 60, 70, 2, -100], "Labels 构造或 Padding 错误！"

        # --- 测试截断后无监督 token 的保护 ---
        try:
            build_sft_data(prompt, [40], pad_id=pad_id, eos_id=eos_id, max_len=3)
            raise AssertionError("截断后没有 response token 时应该报错")
        except ValueError:
            pass

        # --- 测试 Loss 计算 ---
        batch_size = 1
        vocab_size = 100
        logits = torch.randn(batch_size, max_len, vocab_size)

        # 手动让它预测准确：logits[t] 预测 labels[t+1]
        logits[0, 2, 40] = 50.0
        logits[0, 3, 50] = 50.0
        logits[0, 4, 60] = 50.0
        logits[0, 5, 70] = 50.0
        logits[0, 6, 2] = 50.0

        labels_batch = labels.unsqueeze(0)
        attention_batch = attention_mask.unsqueeze(0)
        loss = compute_sft_loss(logits, labels_batch, attention_batch)

        assert loss.item() < 0.01, f"Loss 异常偏大，可能包含了 Prompt 或 Padding 的计算！Loss = {loss.item()}"

        print("\n✅ All Tests Passed! SFT 数据与 loss 对齐逻辑实现正确。")

    except NotImplementedError:
        print("请先完成 TODO 部分的代码！")
        raise
    except (AttributeError, NameError, TypeError, ValueError) as e:
        print("代码可能未完成，导致变量未定义" if isinstance(e, NameError) else "代码可能未完成，导致了类型错误")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise NotImplementedError("请先完成 TODO 部分的代码！") from e
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        raise

test_sft_pipeline()

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
def build_sft_data(
    prompt_ids: list[int],
    response_ids: list[int],
    pad_id: int = 0,
    eos_id: int | None = None,
    max_len: int = 16,
    min_response_tokens: int = 1,
):
    response_with_eos = response_ids + ([] if eos_id is None else [eos_id])
    input_ids = prompt_ids + response_with_eos

    # TODO 1: 构造 labels
    labels = [-100] * len(prompt_ids) + response_with_eos

    # TODO 2: 截断与有效监督检查
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    valid_supervised = sum(label != -100 for label in labels)
    if valid_supervised < min_response_tokens:
        raise ValueError("截断后没有足够的 response token 参与监督，请调大 max_len 或过滤该样本。")

    # TODO 3: attention mask 与填充
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None = None):
    # 预测位置向左对齐一位，对应 next-token prediction。
    # TODO 4: 实现 Shift 错位对齐
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        shift_labels = shift_labels.masked_fill(shift_attention_mask == 0, -100)

    # TODO 5: 检查有效监督 token 并计算交叉熵
    if not torch.any(shift_labels != -100):
        raise ValueError("当前 batch 没有任何有效监督 token，请检查 labels、padding 或截断策略。")

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)

    return loss

```

### 答案与直觉

- **这一题要解决什么：** 把 SFT 的 prompt/response 数据构造、attention mask 和 next-token loss 对齐成一个最小训练闭环。
- **为什么这样做：** 只让 Response/EOS 参与损失，模型才会学会回答并学会结束；shift 则保证预测和标签一一对应。
- **带走的直觉：** SFT 的关键不是“把序列喂进去”，而是“哪些位置可见、哪些位置该学、哪些位置该忽略”。

**1. TODO 1: 构造 labels**

- **实现方式**：`labels = [-100] * len(prompt_ids) + response_with_eos`
- **核心思想**：Prompt 部分全部设为 -100（忽略），Response 和可选 EOS 保持原 token id。
- **Loss Masking 原理**：PyTorch 的 `CrossEntropyLoss` 中，`ignore_index=-100` 的位置不会产生梯度，也不会计入损失。
- **为什么要 mask Prompt**：SFT 的目标是让模型学会“回答”，而不是“背诵提问”。如果 Prompt 也参与损失计算，模型会浪费容量去记忆人类的提问方式。

**2. TODO 2: 截断与有效监督检查**

- **截断逻辑**：`input_ids = input_ids[:max_len]`，`labels = labels[:max_len]`。
- **监督检查**：截断后至少要保留一个非 `-100` 的 label，否则样本没有训练信号。
- **工程细节**：真实微调里如果 prompt 太长、response 被截没，应该调大 `max_len`、缩短 prompt，或直接过滤样本。

**3. TODO 3: attention mask 与填充**

- **attention mask**：真实 token 设为 `1`，padding 设为 `0`。
- **填充逻辑**：
  - `input_ids` 填充 `pad_id`（通常是 tokenizer 的 pad token）
  - `attention_mask` 填充 `0`
  - `labels` 填充 `-100`（确保 Padding 位置不产生梯度）
- **区别**：`attention_mask` 管模型看不看 padding，`labels=-100` 管 loss 学不学这个位置。

**4. TODO 4: Shift 错位对齐**

- **实现方式**：
  ```python
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  ```
- **自回归原理**：模型用前 $t$ 个 token 预测第 $t+1$ 个 token。
- **对齐逻辑**：
  - `logits[0]` 预测的是 `labels[1]`
  - `logits[1]` 预测的是 `labels[2]`
  - 因此需要切掉 `logits` 的最后一个位置，切掉 `labels` 的第一个位置
- **attention mask 二次保护**：如果传入 `attention_mask`，shift 后的 padding label 会再次被设为 `-100`，避免数据构造错误漏进 loss。

**5. TODO 5: 展平并计算交叉熵**

- **实现方式**：
  ```python
  loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
  shift_logits = shift_logits.view(-1, shift_logits.size(-1))
  shift_labels = shift_labels.view(-1)
  loss = loss_fct(shift_logits, shift_labels)
  ```
- **形状要求**：`CrossEntropyLoss` 期望 logits 形状为 `[N, C]`，labels 形状为 `[N]`。
- **有效监督检查**：如果整个 batch 的 labels 都是 `-100`，loss 没有意义，应该显式报错。
- **数据构造**：真实工程中通常在 tokenizer / DataLoader / collator 里批量生成这三件套，而不是逐条手写。
