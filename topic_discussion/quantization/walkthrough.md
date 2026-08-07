# 量化与压缩深入阅读

## 主故事线

一条完整的量化故事通常这样展开：

`model -> quantization goal -> PTQ/QAT choice -> GPTQ/AWQ/FP8/KV cache quant -> deployment benchmark -> final decision`

它和推理优化不同，核心不是把 token 生成得更快，而是先决定“该不该量化、量化哪一部分、量化到什么程度”。

## 1. 先定义目标

先问清楚目标是什么：

- 是要减少权重显存？
- 是要把推理服务部署到更小的卡上？
- 是要压低 KV cache？
- 还是要在硬件支持下提升吞吐？

这个目标决定你要走 PTQ、QAT、GPTQ、AWQ、FP8 还是 KV cache quant。

## 2. 先判断介入时机

如果要最快落地，优先考虑 PTQ。

如果 PTQ 的误差太大，但还有训练预算，就考虑 QAT。

如果训练已经结束，只想在保精度前提下压缩权重，就看 GPTQ / AWQ。

如果硬件和执行栈都支持，就看 FP8。

如果问题主要在长上下文和服务 cache，就看 KV cache quant。

## 3. 决策要和主线联动

量化很少是单独成立的。它通常会和下面这些主线一起出现：

- 推理优化：量化后是否真的提升 TTFT / TPOT / throughput
- 显存优化：量化后是否真的降低 VRAM 压力
- 训练微调：QAT / QLoRA 是否更适合当前任务
- 算子实现：量化有没有对应的 kernel / Triton 支持

所以，量化不是先有结论再找场景，而是先看场景再选方法。

## 4. 回到项目验证

最终都要回到 benchmark report：

- baseline 和 candidate 用同一 workload 了吗
- 显存节省是否真实
- 速度是否真的变快
- 精度损失是否可接受
- 硬件和部署成本是否匹配

这和 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb) 的思路一致，只是量化专题更关注“压缩策略是否值得”。

## 典型路径

### 路径 1：先压权重

`PTQ -> W8A16 -> GPTQ/AWQ -> 67 -> 66`

先从最容易落地的权重量化开始，再用部署和 benchmark 验证收益。

### 路径 2：训练感知压缩

`QAT -> LoRA/QLoRA -> 60`

当 PTQ 误差太大时，先判断是否需要训练感知的补救路线。

### 路径 3：缓存压缩

`FP8 -> KV cache quant -> 41 -> 66`

当服务侧被 cache 预算卡住时，优先处理缓存表示和调度。

## 阅读建议

- 想先知道量化“是什么”，回到 [量化与压缩专题入口](./intro.md)。
- 想看判断框架，回到 [量化与压缩正文](./casebook.md)。
- 想对照推理收益，再去 [推理优化专题](../inference_optimization/intro.md)。
