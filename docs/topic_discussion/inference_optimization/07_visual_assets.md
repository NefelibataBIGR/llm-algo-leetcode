# 07. Visual Assets | 图册收口

## 页面目标

这一页收口推理优化专题的关键图，方便后续把请求链路、prefill、KV cache、decode、量化和选型串起来。

## 图册职责

`07` 不是装饰页，而是把 `01-06` 的抽象关系压成一组可快速定位问题的图。理想状态下，读者看到图就能先回答两个问题：

- 我的问题落在请求链路的哪一段？
- 我下一步更应该去看 kernel、decode、cache、量化还是 benchmark？

## 建议图册

- 请求链路总图
- prefill / attention kernel 图
- KV cache 生命周期和调度图
- decode 策略对照图
- 量化推理与选型图

## 当前已落地图

### 01 请求链路总图

![Inference request lifecycle](/topic_discussion/inference_optimization/request_lifecycle.svg)

### 02 Prefill / Attention Kernel

![Prefill and attention kernel](/topic_discussion/inference_optimization/prefill_attention.svg)

### 04 KV Cache 生命周期与调度

![KV cache lifecycle and scheduling](/topic_discussion/inference_optimization/kv_cache_scheduling.svg)

### 05 量化推理与部署

![Quantized inference and deployment](/topic_discussion/inference_optimization/quantized_deployment.svg)

### 03 Decode 策略对照图

![Decode strategy comparison](/topic_discussion/inference_optimization/decode_strategies.svg)

### 06 Benchmark / Keep-Tune-Switch 决策图

![Benchmark decision flow](/topic_discussion/inference_optimization/benchmark_decision.svg)

## 建议顺序

建议按“总图 -> 子问题 -> 收口图”的顺序组织：

1. 请求链路总图：对应 `01`
2. prefill / attention 图：对应 `02`
3. decode 策略图：对应 `03`
4. KV cache 生命周期与调度图：对应 `04`
5. 量化推理与部署取舍图：对应 `05`
6. benchmark / keep-tune-switch 决策图：对应 `06`

## 图的风格约束

- 一张图只回答一个问题，不把 prefill、decode、cache、量化全塞在一起。
- 正式资产优先用 `SVG`，避免 PNG 栅格化后的字体和留白失真。
- 图标题尽量直白，例如“长 prompt 为什么让 TTFT 变高”，而不是只写名词。
- 如果一个图只能解释 notebook 内部步骤，不足以帮助跨页定位问题，就不应进入专题图册。

## 使用方式

- 先看总图，再看分图。
- 分图只回答一个问题，不把所有策略堆在一张图里。
- 后续如果补 SVG，就优先挂到 `01-06` 的正文页后面。

## 相关跳转

- 回到 [推理优化专题入口](./intro.md)
- 回到 [推理优化正文](./casebook.md)
- 回到 [推理优化深入阅读](./walkthrough.md)
