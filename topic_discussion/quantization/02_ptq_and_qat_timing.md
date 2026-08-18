# 02. PTQ and QAT Timing | PTQ 与 QAT 的介入时机

## 页面目标

这一页回答的是：什么时候先做 PTQ，什么时候必须把量化误差带回训练过程。

## 问题起点

量化真正的第一道选择，不是 GPTQ 还是 AWQ，而是“量化发生在训练后，还是训练中”。这件事决定了：

- 你能不能用最小成本快速落地；
- 你有没有机会让模型适应量化误差；
- 量化路线会不会和 LoRA / QLoRA、继续训练等动作绑在一起。

## 你要先确认什么

- 你有没有训练预算。
- PTQ 后精度损失是否已经不可接受。
- 你的目标是快速部署，还是尽量保住效果。

## 核心矛盾

PTQ 便宜、快、适合快速验证；QAT 更重，但能让模型在训练时学会适应量化误差。两者的差别，不只是“训练前后”的时间点，而是你愿意把多少复杂度付给训练过程。

## 演化路径

1. 先用 PTQ 建立第一版收益账本。
2. 如果精度掉得还能接受，就继续走部署验证。
3. 如果 PTQ 误差太大，但仍有训练预算，就考虑 QAT 或量化感知微调。
4. 如果你已经在 LoRA / QLoRA 路线上，也要重新判断量化误差是否该被纳入训练。

## 关键取舍

- PTQ 更适合“先跑起来、先对齐收益”。
- QAT 更适合“量化误差已经成为主矛盾”。
- 继续训练并不总比 PTQ 更划算，因为训练成本本身也要算进部署收益。

![PTQ and QAT timing](/topic_discussion/quantization/ptq_qat_timing.svg)

## 文献锚点

- PTQ / QAT 经典资料：理解校准与量化感知训练的基本差异。
- QLoRA 相关资料：理解低比特和训练感知为什么会在微调场景里耦合。

## 对应 Part02

- `25` Quantization W8A16
- `26` QLoRA and 4bit Quantization
- `67` Quantized Inference and Deployment

## 典型阅读入口

- [03 Low-Bit Training Adaptation](./03_low_bit_training_adaptation.md)
- [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md)

## 小结

PTQ 和 QAT 的差别，本质上是“先压后测”还是“先让模型学会接受压缩”。
