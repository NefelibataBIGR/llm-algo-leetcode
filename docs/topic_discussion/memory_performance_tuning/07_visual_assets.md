# 07. Visual Assets | 图册收口

## 页面目标

这一页收口显存优化专题的关键图，方便后续把训练账本、checkpointing / offload、推理 cache、量化预算和最终验证串起来。

## 图册职责

`07` 不是装饰页，而是把 `01-06` 的显存关系压成一组能快速定位问题的图：

- 我的问题是训练显存还是推理显存？
- 是 activation、optimizer state、KV cache，还是量化预算在主导峰值？
- 当前动作是在省驻留、重算、搬运，还是压缩表示？

## 建议图册

- VRAM / memory ledger 总图
- training memory pressure 图
- checkpointing / offload trade-off 图
- KV cache budget 图
- quantization as memory tool 图
- benchmark / keep-tune-switch 决策图

## 当前已落地图

### 01 VRAM / Memory Ledger

![VRAM ledger](/topic_discussion/memory_performance_tuning/vram_ledger.svg)

### 02 Training Memory Pressure

![Training memory pressure](/topic_discussion/memory_performance_tuning/training_memory_pressure.svg)

### 03 Checkpointing / Offload

![Checkpointing and offload trade-off](/topic_discussion/memory_performance_tuning/checkpointing_offload.svg)

### 04 KV Cache Budget

![KV cache budget](/topic_discussion/memory_performance_tuning/kv_cache_budget.svg)

### 05 Quantization as a Memory Tool

![Quantization as a memory tool](/topic_discussion/memory_performance_tuning/quantization_memory_tool.svg)

### 06 Benchmark / Keep-Tune-Switch

![Memory benchmark decision flow](/topic_discussion/memory_performance_tuning/memory_benchmark_decision.svg)

## 建议顺序

1. 账本总图：对应 `01`
2. 训练侧显存压力图：对应 `02`
3. checkpointing / offload 图：对应 `03`
4. 推理 cache 与预算图：对应 `04`
5. 量化作为显存手段图：对应 `05`
6. benchmark / 决策图：对应 `06`

## 图的风格约束

- 一张图只回答一个显存问题，不把训练、推理和部署强行塞进同一张图。
- 正式资产优先用 `SVG`。
- 图标题尽量直接说明“谁在占显存、代价换到哪里去了”。

## 相关跳转

- 回到 [显存优化与性能调优专题入口](./intro.md)
- 回到 [显存优化与性能调优正文](./casebook.md)
- 回到 [显存优化与性能调优深入阅读](./walkthrough.md)
