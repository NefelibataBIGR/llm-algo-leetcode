# 09 Cross Module Comparison

## 页面目标

这一页不是讲新模块，而是把前面的模块页横向拉通，帮助读者把“演进史”转成“结构选择地图”。

## 为什么需要这一页

单独看每一页，你能理解：

- tokenization 怎么变
- norm 怎么变
- attention 怎么变
- RoPE 怎么变
- decoder 怎么变
- FFN / MLP 怎么变
- block 怎么组装
- 代表模型怎么选型

但如果不做横向对照，容易出现两个问题：

- 只记住单页内容，不知道它们之间怎么连接
- 只知道模型名字，不知道模型结构到底改了哪一块

这一页就是用来避免这两个问题的。

## 对照维度

### 维度 1：表示入口

| 模块 | 关注点 |
|:---|:---|
| `tokenization_embedding` | 文本如何变成 token id 和 hidden state |
| `representative_models` | 不同模型的 tokenizer / embedding 有什么工程差异 |

### 维度 2：数值稳定性

| 模块 | 关注点 |
|:---|:---|
| `norm_evolution` | LayerNorm、RMSNorm、Pre-Norm 的稳定性取舍 |
| `block_residual_path` | norm 和 residual 如何一起决定训练稳定性 |

### 维度 3：上下文建模

| 模块 | 关注点 |
|:---|:---|
| `attention_evolution` | attention 如何在表达能力和成本之间取舍 |
| `rope_position_encoding` | 位置关系如何进入 attention |

### 维度 4：单 token 变换

| 模块 | 关注点 |
|:---|:---|
| `mlp_ffn_evolution` | token 内部的非线性如何增强 |
| `block_residual_path` | FFN 如何嵌入 block 并和 residual 协同 |

### 维度 5：结构落地

| 模块 | 关注点 |
|:---|:---|
| `transformer_decoder` | decoder-only 为什么成为主流主干 |
| `representative_models` | 各模型如何在 decoder-only 上做差异化设计 |

## 一组典型对照

### 对照 1：LLaMA

- `tokenization_embedding`：标准 subword 表示入口
- `norm_evolution`：RMSNorm
- `attention_evolution`：MHA / GQA 路线
- `rope_position_encoding`：RoPE
- `mlp_ffn_evolution`：SwiGLU
- `transformer_decoder`：decoder-only 主干
- `block_residual_path`：现代 dense block 参照系

### 对照 2：Mistral

- 更适合看 `attention_evolution`
- 和 `rope_position_encoding`、`transformer_decoder` 结合得更紧
- 体现长上下文和系统成本的折中

### 对照 3：Qwen

- 更适合看 `tokenization_embedding`
- 也适合看 `representative_models`
- 强调工程可用性和多语言覆盖

### 对照 4：DeepSeek

- 更适合看 `attention_evolution`
- 也适合看 `representative_models`
- 体现结构重构和效率优化的前沿方向

## 读法建议

如果你想查某个问题，建议按下面顺序回跳：

- 想看“输入怎么变成模型表示” -> `tokenization_embedding`
- 想看“为什么训练更稳” -> `norm_evolution` + `block_residual_path`
- 想看“长上下文为什么慢” -> `attention_evolution` + `rope_position_encoding`
- 想看“为什么 decoder-only 成主流” -> `transformer_decoder`
- 想看“某个大模型到底改了什么” -> `representative_models`

## 可视化提示

建议做一张横向矩阵图：

- 行：专题单页
- 列：表示入口、稳定性、上下文、单 token 变换、结构落地、模型案例

这样读者可以快速定位每页在整体知识图谱里的位置。
