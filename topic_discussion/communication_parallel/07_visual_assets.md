# 07. Visual Assets | 图册收口

## 页面目标

这一页收口通信并行专题的关键图，方便后续把通信原语、状态切分、pipeline / tensor parallel、expert parallel 和最终决策串起来。

## 图册职责

`07` 不是装饰页，而是把 `01-06` 的并行关系压成一组能快速回答下面问题的图：

- 我现在的问题是同步、显存、bubble，还是路由热点？
- 当前应该先看数据并行、ZeRO、Pipeline / Tensor Parallel，还是 Expert Parallel？
- 这条并行路线最终服务的是显存扩展、吞吐扩展，还是模型规模扩展？

## 建议图册

- 为什么系统会走向并行的总图
- 数据并行与同步图
- ZeRO / 状态切分图
- Pipeline / Tensor Parallel 对照图
- Expert Parallel 与热点图
- benchmark / keep-tune-switch 决策图

## 当前已落地图

### 01 为什么系统会走向并行

![Why systems move to parallelism](/topic_discussion/communication_parallel/parallel_overview.svg)

### 02 数据并行与同步

![Data parallelism and synchronization](/topic_discussion/communication_parallel/data_parallel_sync.svg)

### 03 ZeRO / 状态切分

![State sharding and ZeRO](/topic_discussion/communication_parallel/zero_sharding.svg)

### 04 Pipeline / Tensor Parallel

![Pipeline and tensor parallel](/topic_discussion/communication_parallel/pipeline_tensor_parallel.svg)

### 05 Expert Parallel / Hotspots

![Expert parallel and communication hotspots](/topic_discussion/communication_parallel/expert_parallel_hotspots.svg)

### 06 Benchmark / Keep-Tune-Switch

![Parallel benchmark decision flow](/topic_discussion/communication_parallel/parallel_benchmark_decision.svg)

## 建议顺序

1. 并行起点与约束：对应 `01`
2. 数据并行与同步：对应 `02`
3. 状态切分与 ZeRO：对应 `03`
4. Pipeline / Tensor Parallel：对应 `04`
5. Expert Parallel 与热点：对应 `05`
6. benchmark 与决策：对应 `06`

## 图的风格约束

- 一张图只回答一个并行问题，不把所有切分方式塞进同一张图。
- 正式资产优先用 `SVG`。
- 图标题尽量直接说明“切的是什么、代价换到哪里去了”。

## 相关跳转

- 回到 [通信与并行专题入口](./intro.md)
- 回到 [通信与并行正文](./casebook.md)
- 回到 [通信与并行深入阅读](./walkthrough.md)
