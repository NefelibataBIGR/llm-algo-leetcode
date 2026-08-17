# 04 通信等待与 Overlap

## 页面目标

这一页负责把 profiling 的视角扩到多卡：谁在等、哪里没 overlap、为什么理论并行收益没有兑现。

## 问题起点

多卡变慢时，常见误判是：

- 以为算子本身慢
- 以为 GPU 利用率低就是单卡问题

但真实情况往往是：

- 同步等待高
- overlap 没生效
- communication hotspot 把收益吃掉了

## 关键观察点

- communication wait
- all-reduce / all-gather hotspots
- pipeline bubble
- overlap 是否存在

## 可视化入口

![Communication Wait and Overlap Map](/topic_discussion/profiling/communication_overlap_map.svg)

## 对应 Part

- `46 Communication Profiling with NCCL`
- `79 Distributed Parallel Benchmark`
- `74 Profiling Driven End-to-End Optimization`

## 小结

多卡 profiling 的关键不是“看更多 trace”，而是解释为什么通信和等待把理想收益吃掉了。
