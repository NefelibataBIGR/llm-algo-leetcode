# 显存优化与性能调优深入阅读

假设你接手的是同一个系统：训练阶段在中后段开始 OOM，勉强把训练跑完以后，服务侧又发现长上下文和多轮对话会把 KV cache 顶高，最终不是 batch 上不去，就是延迟和吞吐一起变差。

这条线最重要的是按暴露顺序判断：训练先在哪一侧失控，推理又在哪一侧顶住预算，最后哪些方案只是止血，哪些方案真的值得保留。

## 第一段：训练中后段开始 OOM

故事通常从一个很典型的症状开始：前几个 step 都正常，loss 也没问题，但训练跑到中后段时显存开始持续抬高，最后在某个 step 直接 OOM。第一反应往往是缩 batch，但这通常只是止血动作，不是判断结论。更稳的做法是先沿训练侧显存链路排一遍：

- Part 02 [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- Part 02 [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
- Part 02 [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)

这一段真正要拆开的，是 `effective batch`、`activation`、`checkpointing` 和 `offload`。很多时候 batch 只是把 activation、重算和搬运成本一并放大了，而不是唯一矛盾本身。

## 第二段：训练能跑了，但推理还是装不下

训练阶段止血以后，第二个问题往往出现在部署或服务验证阶段。模型能加载，但只要上下文拉长、并发上去，显存又开始被 KV cache 顶满。这时要切到推理侧显存链路：

- Part 02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
- Part 02 [24 SGLang RadixAttention](../../02_PyTorch_Algorithms/24_SGLang_RadixAttention.ipynb)
- Part 02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)
- Part 02 [37 KV Cache Scheduling](../../02_PyTorch_Algorithms/37_KV_Cache_Scheduling.ipynb)
- Part 02 [41 FP8 and KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb)
- Part 02 [67 Quantized Inference and Deployment](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb)

这里的核心不是“为什么慢”，而是“为什么装不下”。要先分清 cache 增长是不是请求形态的自然结果，prefix reuse 和 paging 是否足够，KV cache quantization 是否值得引入。

## 第三段：账本和实测开始打架

走到这一步，团队通常会碰到第三类问题：理论上算出来应该够，实测却还是很紧；或者峰值显存降下来了，但 benchmark 没好多少。这时就不能只看局部收益，而要把账本和实测证据对齐：

- Part 01 [06 VRAM Calculation and ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
- Part 01 [13 Profiling and Bottleneck Analysis](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
- Part 02 [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)

这一段真正要回答的是：理论账本里有没有漏掉临时 buffer、碎片或流程开销；训练侧峰值下降是不是只是把时间转移到了别处；推理侧 cache 压缩是不是只是把显存问题换成了延迟问题。

## 第四段：把候选方案放回同一张对比表

到了真正做决策的时候，不能只写“试过 checkpointing、offload、KV quant”，而要把它们放回同一张 baseline / candidate 对比表里。更稳的收口方式通常是：`baseline` 保留原始训练和服务配置，`candidate A` 先用 checkpointing / batch 调整止血，`candidate B` 再引入 offload，`candidate C` 用 paging / prefix reuse / KV quant 把 cache 压回预算。真正的判断不该停在“省了显存”，而要落成 `accept / tune / reject`。

## 最终结论长什么样

把这条故事走完以后，一个更像真实交付的结论通常不是“我们用了某个省显存技巧”，而是：训练中后段 OOM 的主要矛盾在 activation 与 effective batch，推理侧的主要约束转到了 KV cache，最终被接受的不是某个单点动作，而是一组训练侧和推理侧都站得住的配置组合。
