# 反向传播与训练机制正文

## 页面目标

这页把专题拆成五个机制块。它不是章节索引，而是每个块的写作骨架和排障清单。

## 机制块

### 01 反向传播总览与计算图

- backward 为什么会决定训练能不能跑起来
- 计算图如何组织梯度回传
- 哪些状态必须保留，哪些可以重算

### 02 Autograd 与 Attention Backward

- `grad_fn` / `saved_tensors` / 自定义 `autograd.Function`
- attention 的 `dV -> dP -> dS -> dQ / dK`
- 为什么 attention backward 同时是数学问题和访存问题

### 03 Loss Backward、标签对齐与显存账本

- `mask / shift / ignore_index`
- prompt / response 的监督边界
- activation、参数、梯度、optimizer state 的显存账本

### 04 Checkpointing 与 Offload

- 重算换显存
- 搬运换显存
- 两者的边界、代价和适用场景

### 05 梯度累积、训练闭环与 Profiling

- `micro-batch / accumulation steps / effective batch`
- backward / step / profiling 的闭环
- 什么时候先看调度，什么时候先看显存

## 常见检查项

| 检查项 | 你要确认什么 | 常见问题 |
|:---|:---|:---|
| 计算图 | 梯度沿什么路径回传 | 只会写 forward，不会解释 backward |
| Autograd | `grad_fn`、`saved_tensors` 是否看得懂 | 把 API 当成机制本身 |
| Attention backward | `Q / K / V / P` 的反向顺序是否清楚 | 只记公式，不看保存点 |
| 标签对齐 | 监督区间是否正确 | loss 算对了，标签口径错了 |
| 显存账本 | 训练显存到底被什么占住 | 只盯 activation，忽略参数和 optimizer state |
| Checkpointing / Offload | 是重算还是搬运 | 混淆两类代价模型 |
| 梯度累积 | backward 次数和 step 次数是否统一 | 训练节奏错位 |
| Profiling | 优化前后收益是否可复现 | 只看感觉，不看 baseline |

## 机制对照

| 机制 | 解决什么 | 代价是什么 | 适合什么时候用 |
|:---|:---|:---|:---|
| 反向传播与计算图 | 让梯度能正确回传 | 需要保存或重算状态 | 所有训练任务 |
| Autograd 与 attention backward | 把机制落到 PyTorch / 算子上 | 依赖中间量保存 | 需要理解实现细节时 |
| Loss 对齐与显存账本 | 监督口径 + 训练显存预算 | 需要更细的标签和状态管理 | SFT / chat 训练 |
| Checkpointing / Offload | 省 GPU 显存 | 重算或搬运代价 | 显存紧张时 |
| 梯度累积 + Profiling | 把训练放回闭环验证 | 节奏更长、观测成本更高 | 需要稳定训练和定位瓶颈时 |

## 任务映射

| Task | 关注点 |
|:---|:---|
| Task1 | 反向传播与计算图 |
| Task2 | Autograd 与 attention backward |
| Task3 | Loss 对齐与显存账本 |
| Task4 | Checkpointing 与 offload |
| Task5 | 梯度累积与 profiling |

## 相关跳转

- 想看完整路线，回到 [反向传播与训练机制专题入口](./intro.md)。
- 想看连续故事线，去 [反向传播与训练机制深入阅读](./walkthrough.md)。
- 想看训练微调闭环，去 [训练微调闭环专题](../fine_tuning_training/intro.md)。
- 想看显存调优，去 [显存优化与性能调优专题](../memory_performance_tuning/intro.md)。
