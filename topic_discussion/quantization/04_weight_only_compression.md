# 04. Weight-Only Compression | 权重量化与后训练压缩

## 页面目标

这一页回答的是：GPTQ、AWQ 和 weight-only 路线在解决什么问题，为什么它们经常成为后训练量化的主轴。

## 问题起点

很多部署场景的首要问题不是激活，而是权重驻留太大。于是，最先被问到的问题往往是：

- 能不能先把权重压下来，让模型装进目标卡；
- 压下来以后，精度损失能不能控制在可接受范围内。

GPTQ / AWQ 就是在这个问题上出现的。

## 你要先确认什么

- 权重驻留是不是预算主因。
- 你更看重极限压缩率，还是更稳的精度保持。
- 你有没有继续训练预算；如果没有，这一页通常就是主轴。

## 核心矛盾

权重量化的核心矛盾是：低比特能显著降低模型驻留成本，但某些层、通道和矩阵对误差异常敏感。GPTQ / AWQ 的存在，就是为了在后训练阶段尽量保住这些敏感位置。

## 演化路径

1. 最基础的是 W8A16 这类易落地的权重量化。
2. 如果压缩还不够或精度退化太大，再看 GPTQ / AWQ。
3. 最后再回到部署和 benchmark，确认收益是否真的匹配硬件和服务目标。

## 关键取舍

- GPTQ 更偏误差补偿和近似最优重建。
- AWQ 更偏激活感知，优先保护敏感通道。
- 两者都属于后训练路线，但侧重点不同，不是简单替换关系。

![Weight-only compression](/topic_discussion/quantization/weight_only_compression.svg)

## 文献锚点

- GPTQ：理解后训练权重量化为何能用误差补偿稳住精度。
- AWQ：理解为什么激活感知会影响权重量化的保精度能力。

## 对应 Part02

- `25` Quantization W8A16
- `40` GPTQ and AWQ Weight Quantization
- `67` Quantized Inference and Deployment

## 典型阅读入口

- [01 Quantization Object and Error](./01_quantization_object_and_error.md)
- [06 Deployment and Benchmark Decision](./06_deployment_and_benchmark_decision.md)

## 小结

权重量化的主问题不是“能不能压”，而是“压完以后能不能还值这个部署收益”。
