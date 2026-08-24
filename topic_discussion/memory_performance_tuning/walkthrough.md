# 显存优化与性能调优深入阅读

假设你接手的是同一个系统：训练阶段在中后段开始 OOM，勉强把训练跑完以后，服务侧又发现长上下文和多轮对话会把 KV cache 顶高，最终不是 batch 上不去，就是延迟和吞吐一起变差。

这条线最重要的是按暴露顺序判断：训练先在哪一侧失控，推理又在哪一侧顶住预算，最后哪些方案只是止血，哪些方案真的值得保留。

主项目线分成“训练侧决策”和“最终收口”两段：`73` 建立训练基线，`76` 比较 checkpoint / offload / hybrid，`75` 形成训练侧预算决策，`74` 再用 profiling 对显存优化方案做端到端最终验证。Task 4 和 Task 5 的推理、量化内容是扩展分支，不是所有学习者的硬性前置。

## 第一段：训练中后段开始 OOM

故事通常从一个很典型的症状开始：前几个 step 都正常，loss 也没问题，但训练跑到中后段时显存开始持续抬高，最后在某个 step 直接 OOM。第一反应往往是缩 batch，但这通常只是止血动作，不是判断结论。更稳的做法是先沿训练侧显存链路排一遍：

- Part 02 [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.ipynb)
- Part 02 [19 Activation Checkpointing and Activation Offload](../../02_PyTorch_Algorithms/19_Activation_Checkpointing_and_Activation_Offload.ipynb)
- Part 02 [42 Activation Offload](../../02_PyTorch_Algorithms/42_Activation_Offload.ipynb)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
- Part 02 [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
- Part 02 [75 Memory Budget Compression Project](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)

这一段真正要拆开的，是 `effective batch`、`activation`、`checkpointing` 和 `offload`。核心路径先完成 `73 -> 76 checkpoint -> 75`；`offload / hybrid` 和更高压力 workload 属于训练侧扩展。很多时候 batch 只是把 activation、重算和搬运成本一并放大了，而不是唯一矛盾本身。

## 第二段：训练能跑了，但推理还是装不下

训练阶段止血以后，第二个问题往往出现在部署或服务验证阶段。模型能加载，但只要上下文拉长、并发上去，显存又开始被 KV cache 顶满。这时要切到推理侧显存链路：

- Part 02 [22 vLLM PagedAttention](../../02_PyTorch_Algorithms/22_vLLM_PagedAttention.ipynb)
- Part 02 [34 Prefix Caching and Chunked Prefill](../../02_PyTorch_Algorithms/34_Prefix_Caching_and_Chunked_Prefill.ipynb)
- Part 02 [66 Inference Performance Comparison](../../02_PyTorch_Algorithms/66_Inference_Performance_Comparison.ipynb)

核心路径先看 `22 -> 34`，再用 `66` 完成单 backend 最小验证；`24 RadixAttention`、`37 KV Cache Scheduling`、`41 KV Cache Quantization` 和 `67` 的真实量化 backend 部署属于扩展路径。

这里的核心不是“为什么慢”，而是“为什么装不下”。要先分清 cache 增长是不是请求形态的自然结果，prefix reuse 和 paging 是否足够，KV cache quantization 是否值得引入。

量化是另一条显存扩展分支：核心先看 [21 量化理论](../../01_Hardware_Math_and_Systems/21_Quantization_Theory_and_INT4_INT8.ipynb)、[25 W8A16](../../02_PyTorch_Algorithms/25_Quantization_W8A16.ipynb) 和 [67 量化推理与部署](../../02_PyTorch_Algorithms/67_Quantized_Inference_and_Deployment.ipynb) 的本地加载；再按需进入 [40 GPTQ / AWQ](../../02_PyTorch_Algorithms/40_GPTQ_and_AWQ_Weight_Quantization.ipynb) 和 [41 FP8 / KV Cache Quantization](../../02_PyTorch_Algorithms/41_FP8_and_KV_Cache_Quantization.ipynb) 的真实 backend 扩展。这里要判断的是：量化省下来的显存是否换来了更长上下文、更大 batch 或更高并发，而不是只看权重文件变小。

## 第三段：账本和实测开始打架

走到这一步，团队通常会碰到第三类问题：理论上算出来应该够，实测却还是很紧；或者峰值显存降下来了，但 benchmark 没好多少。这时就不能只看局部收益，而要把账本和实测证据对齐：

- Part 01 [06 VRAM Calculation and ZeRO](../../01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.ipynb)
- Part 01 [13 Profiling and Bottleneck Analysis](../../01_Hardware_Math_and_Systems/13_Profiling_and_Bottleneck_Analysis.ipynb)
- Part 02 [73 Training Performance Analysis](../../02_PyTorch_Algorithms/73_Training_Performance_Analysis.ipynb)
- Part 02 [76 Activation / Checkpoint / Offload Benchmark](../../02_PyTorch_Algorithms/76_Activation_Checkpoint_Offload_Benchmark.ipynb)
- Part 02 [75 Memory Budget Compression Project](../../02_PyTorch_Algorithms/75_Memory_Budget_Compression_Project.ipynb)
- Part 02 [74 Profiling Driven End-to-End Optimization](../../02_PyTorch_Algorithms/74_Profiling_Driven_End_to_End_Optimization.ipynb)

这一段真正要回答的是：理论账本里有没有漏掉临时 buffer、碎片或流程开销；训练侧峰值下降是不是只是把时间转移到了别处；推理侧 cache 压缩是不是只是把显存问题换成了延迟问题。`74` 不替代 `75` 的训练侧预算决策，而是负责最后的 profiling 和端到端验证。

## 第四段：把候选方案放回同一张对比表

到了真正做决策的时候，不能只写“试过 checkpointing、offload、KV quant”，而要把它们放回同一张 baseline / candidate 对比表里。训练侧先由 `75` 输出 `accept / tune / reject`，再由 `74` 检查候选方案在端到端 workload 下是否仍然成立：`baseline` 保留原始配置，`candidate A` 先用 checkpointing 止血，`candidate B` 再引入 offload，推理扩展中再比较 paging / prefix reuse / KV quant。真正的判断不该停在“省了显存”。

## 最终结论长什么样

把这条故事走完以后，一个更像真实交付的结论通常不是“我们用了某个省显存技巧”，而是：训练中后段 OOM 的主要矛盾在 activation 与 effective batch，训练侧方案经过 `73 -> 76 -> 75` 做出预算判断，随后由 `74` 验证端到端代价；如果继续进入推理侧，再说明 KV Cache、量化或 backend 配置是否改变了可部署边界。
