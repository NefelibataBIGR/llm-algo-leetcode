# 01 为什么 Profiling 值得单独成章

## 页面目标

这一页先解释 profiling 的目标：不是“多看图”，而是把一个性能猜测变成证据链。

## 问题起点

训练和推理里的“慢”，往往有很多表象：

- step time 变长
- TTFT 变高
- 多卡扩展不稳
- 显存降了但吞吐也掉了

如果没有 profiling，很多结论都停留在“怀疑某个模块慢”，而不是“证明确实慢在这里”。

## 核心矛盾

profiling 想得到更可靠的判断，但代价是：

- 采集会更复杂
- 图和表会更多
- 更容易被局部热点误导

所以这条专题的关键，不是把工具列全，而是教人怎样建立一条可靠的诊断链。

## 可视化入口

![Profiling Evidence Chain](/topic_discussion/profiling/profiling_evidence_chain.svg)

## 对应 Part

- `0E / 17 / 20`：profiling 的入门和前置桥。
- `74 / 79 / 46`：profiling 在真实训练、分布式和通信场景里的落点。

## 小结

profiling 的价值在于：它把“感觉慢”变成“证据证明慢在哪里”。
