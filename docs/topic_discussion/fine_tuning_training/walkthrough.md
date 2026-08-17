# 监督微调（SFT）闭环深入阅读

假设你接手的是一个新的监督微调任务：已经有一批 `prompt / response` 数据，目标不是证明模型“能训练”，而是把这批数据走成一个可验证、可交付的 LoRA 微调闭环。

这条线最重要的是按暴露顺序判断：样本先在哪里出问题，loss 口径是否成立，LoRA 是否真的挂对了层，训练控制有没有把实验做歪，最后产出的东西能不能真正交付。

## 第一段：先确认样本能不能进入训练

故事通常从一条具体样本开始。这个阶段最常见的问题不是模型本身，而是数据格式就已经不稳：prompt / response 是否完整，是否需要 system / user / assistant 结构，有没有空 response、重复样本或脏格式。真正进入训练前，要先把它压成 `input_ids / attention_mask / labels` 这组三件套；如果这里没对齐，后面的 loss 曲线大概率没有解释力。

## 第二段：loss 口径成立以后，再看 LoRA 挂载

数据能进训练以后，第二个问题通常不是“有没有 LoRA”，而是 LoRA 到底挂在什么地方、训练参数占比是否合理。这一段要沿下面的线去看：

- Part02 [09 SFT Training Loop](../../02_PyTorch_Algorithms/09_SFT_Training_Loop.md)
- Part02 [10 LoRA Tutorial](../../02_PyTorch_Algorithms/10_LoRA_Tutorial.md)
- Part02 [26 QLoRA and 4bit Quantization](../../02_PyTorch_Algorithms/26_QLoRA_and_4bit_Quantization.md)

这里真正想确认的是：target modules 是否挂在真正重要的层，`r / alpha / dropout` 是否合理，可训练参数占比是不是符合预期。这一段最常见的误判是：只要用了 LoRA，就默认已经做了有效微调；真实情况是，挂错层或参数比例失衡时，训练能跑，但结果可能几乎没变化。

## 第三段：训练能跑，不代表训练控制口径正确

LoRA 挂进去以后，第三个问题经常出在训练控制。loss 在降，但实验结论并不可信，常见原因是 scheduler 计数口径不对、optimizer step 和 accumulation 不一致、effective batch 和你以为的不一样。这一段主要沿下面的线去看：

- Part02 [11 LR Schedulers WSD Cosine](../../02_PyTorch_Algorithms/11_LR_Schedulers_WSD_Cosine.md)
- Part02 [12 Gradient Accumulation](../../02_PyTorch_Algorithms/12_Gradient_Accumulation.md)
- Part02 [13 End-to-End Fine-Tuning Experiment](../../02_PyTorch_Algorithms/13_End_to_End_Fine_Tuning_Experiment.md)

这里最容易出现的误判是：只要训练没报错，训练控制就是对的。实际上很多“看起来能跑”的实验，最后只是 step 口径混了。

## 第四段：实验跑通以后，先看结论是不是站得住

训练结束以后，最危险的动作是只看 train loss。真正需要回答的是：val loss 有没有同步变化，生成样例有没有真的变好，显存和速度代价是不是在可接受范围内，这次配置值不值得保留。这一步本质上是在把“机制正确”推进到“实验成立”。

## 第五段：最后回到项目收口

真正的闭环结束点不是“训练完成”，而是项目页：

- Part02 [32 Data Engineering for SFT](../../02_PyTorch_Algorithms/32_Data_Engineering_for_SFT.md)
- Part02 [33 Fine Tuning Readiness](../../02_PyTorch_Algorithms/33_Fine_Tuning_Readiness.md)
- Part02 [60 LoRA Fine-Tuning Project](../../02_PyTorch_Algorithms/60_LoRA_Fine_Tuning_Project.md)

这时要回答的已经不是“能不能训”，而是：数据是否可信，loss 口径是否正确，adapter、tokenizer、config 是否可交付，最终结果是否值得采用。基础闭环跑通以后，再进入 `26 -> 31 -> 15 -> 16` 这些分支，看小显存微调、LoRA 变体和对齐后续路线。把这条故事走完以后，一个更像真实交付的结论通常不是“我们完成了一次 SFT”，而是：数据口径正确、LoRA 挂载合理、训练控制一致、实验结果站得住，最终产出的 adapter 和报告可以被别人复现和采用。
