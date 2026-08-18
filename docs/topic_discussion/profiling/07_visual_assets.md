# 07 Visual Assets

## 页面目标

这页负责收口 profiling 专题的图册资产。第一阶段先固定图的职责，后续再逐步补正式 SVG。

## 图册顺序

1. `profiling_evidence_chain`
- 从问题提出到采证、归因、验证、行动的总图

![Profiling Evidence Chain](/topic_discussion/profiling/profiling_evidence_chain.svg)

2. `time_breakdown_trace`
- operator / kernel / wait / launch 的时间拆分图

![Time Breakdown and Trace Reading](/topic_discussion/profiling/time_breakdown_trace.svg)

3. `memory_timeline_diagnosis`
- memory timeline 和 residency 的诊断图

![Memory Timeline Diagnosis](/topic_discussion/profiling/memory_timeline_diagnosis.svg)

4. `communication_overlap_map`
- 多卡等待与 overlap 关系图

![Communication Wait and Overlap Map](/topic_discussion/profiling/communication_overlap_map.svg)

5. `benchmark_validation_board`
- before / after、波动和回归设计图

![Benchmark Validation Board](/topic_discussion/profiling/benchmark_validation_board.svg)

6. `action_decision_board`
- keep observing / inspect / optimize / revert 的决策图

![Diagnosis and Action Decision](/topic_discussion/profiling/action_decision_board.svg)

## 当前状态

第一批和第二批图已补齐，当前图册已覆盖 `01-06` 的主要入口。
