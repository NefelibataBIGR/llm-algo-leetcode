# 05. Quantized Inference and Deployment | 量化推理与部署

## 页面目标

这一页回答的是：权重、激活和 KV cache 量化分别改变什么成本，以及什么时候值得切换。

## 问题起点

量化常常被误写成“显存不够时的默认答案”。但推理场景下，量化真正要回答的是更具体的问题：

- 我是被权重驻留卡住，还是被带宽卡住，还是被 KV cache 顶住？
- 我更敏感的是 TTFT、TPOT，还是 throughput / cost？
- 这个 backend 和 deployment 栈对低比特支持到什么程度？

只有先把约束说清楚，量化才是选型动作，而不是默认操作。

## 你要先确认什么

- 你要优化的是显存、带宽还是部署成本。
- 量化后 TTFT、TPOT、throughput 是否可接受。
- 线上服务是否更敏感延迟还是吞吐。

## 核心矛盾

量化的核心矛盾是：低比特表示能省显存和带宽，但会引入精度误差、kernel 兼容性和部署复杂度。它从来不是“免费更快”，只能说是在某些 workload 下值得交换。

## 演化路径

量化不是一个统一动作，而是分别作用在不同部位。

1. 权重量化降低模型驻留成本。
2. 激活量化改变中间计算和带宽压力。
3. KV cache 量化直接影响长上下文和并发边界。
4. GPTQ / AWQ 更偏权重压缩。
5. FP8 / KV cache quant 更偏部署侧平衡。

## 关键取舍

- 权重量化更像“把模型装下或提高 batch 的第一步”。
- `GPTQ / AWQ` 更偏离线权重压缩与精度权衡。
- `FP8` 更偏端到端部署栈是否愿意为低精度继续优化 kernel。
- `KV cache quantization` 更像推理侧资源手段，尤其适合长上下文和高并发。

最终判断要回到服务目标：

- 在线交互更怕 TTFT / TPOT 退化；
- 离线批处理更愿意为 throughput / cost 接受一定延迟变化。

![Quantized inference and deployment](/topic_discussion/inference_optimization/quantized_deployment.svg)

## 文献锚点

- GPTQ：帮助理解离线权重量化如何用近似最优重建减少精度损失。
- AWQ：帮助理解按激活感知选择量化尺度的动机。
- FP8 / KV cache quantization 相关资料：帮助理解部署侧怎样平衡吞吐、显存和质量。

## 常见误区

- 看到 peak memory 降了就认为方案一定更好。
- 不分在线交互和离线批处理，直接比较量化收益。
- 把推理量化和训练量化混在一起看。

## 对应 Part 02

- `25` Quantization W8A16
- `40` GPTQ and AWQ Weight Quantization
- `41` FP8 and KV Cache Quantization
- `67` Quantized Inference and Deployment

## 经典阅读入口

- [21 Quantization Theory and INT4 INT8](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.md)
- [25 Quantization W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.md)
- [40 GPTQ and AWQ Weight Quantization](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.md)
- [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.md)
- [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.md)

## 相关跳转

- 看 `01`，先统一指标口径。
- 看 `04`，确认 cache 是否已经是硬约束。
- 看 `06`，把量化和其他候选方案一起比较。

## 本节要点

量化是推理优化的候选方案之一，不是默认答案；最终仍要回到 workload 和服务目标。
