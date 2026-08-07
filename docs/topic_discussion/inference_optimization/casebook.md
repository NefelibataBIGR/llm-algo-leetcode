# 推理优化正文

## 页面目标

这页把推理优化专题沉淀成可操作的判断框架。它不重复每个 notebook 的机制细节，而是回答：

- 看到 TTFT、TPOT、throughput、peak memory 后怎么判断瓶颈？
- 长 prompt、decode 慢、cache 涨、量化收益不稳定分别应该看哪条线？
- 最后怎么把判断收束到 `66` 的推理性能对比项目？

## 指标口径

| 指标 | 含义 | 主要关联 |
|:---|:---|:---|
| `TTFT` | Time To First Token，首 token 延迟，通常近似 prefill latency | 长 prompt、attention kernel、chunked prefill |
| `TPOT` | Time Per Output Token，decode 阶段平均每 token 延迟 | decode loop、KV cache 读写、调度 |
| `throughput` | 单位时间生成 token 数，通常看 generated tokens/s | batching、调度、推测解码、多 token 解码 |
| `peak memory` | 推理峰值显存 | 权重、KV cache、batch size、量化 |
| `prefill_share` | prefill 占总耗时比例 | attention 访存、prompt length |
| `decode_share` | decode 占总耗时比例 | KV cache、sampling、decode scheduling |

`66` 已经把这些指标放进项目模板：先固定 workload，再拆 prefill/decode，再比较 baseline 和 candidate。

## 瓶颈诊断

| 瓶颈 | 典型信号 | 优先阅读 | 常见动作 |
|:---|:---|:---|:---|
| `prefill-bound` | `TTFT` 高、`prefill_share` 高、长 prompt 变慢明显 | Task2 + Task4 | FlashAttention、chunked prefill、prefix caching |
| `decode-bound` | `TPOT` 高、`decode_share` 高、并发下 token 产出慢 | Task3 + Task4 | speculative decoding、multi-token decoding、decode scheduling |
| `memory-bound` | peak memory 接近预算，batch 或上下文上不去 | Task4 + Task5 | PagedAttention、KV cache scheduling、KV cache quantization |
| `balanced` | 没有明显单点瓶颈 | Task6 + Profiling 专题 | 保持 baseline 或继续做更细粒度 profiling |

显存接近预算时，优先按 `memory-bound` 处理。显存是硬约束；即使 decode 占比高，如果 KV cache 已经顶到预算，继续扩大 batch 或上下文都不可靠。

## Task 对照

| Task | 关键词 | 适合解决的问题 |
|:---|:---|:---|
| Task1 | Attention、GQA、Block | 我知道推理在跑模型，但不知道主要结构成本在哪里 |
| Task2 | HBM、SRAM、tiling、online softmax | 长 prompt / attention 太慢，想理解 FlashAttention 为什么有效 |
| Task3 | sampling、speculative、multi-token | 生成阶段 token 产出慢，想减少 decode 循环成本 |
| Task4 | KV cache、prefix、paging、scheduling | 服务侧吞吐上不去，cache 复用和请求排布有问题 |
| Task5 | W8A16、GPTQ、AWQ、FP8、KV cache quant | 显存或带宽受限，想用量化换部署收益 |
| Task6 | TTFT、TPOT、throughput、decision | 需要用 benchmark 判断方案该 keep、tune 还是 switch |

## 典型案例

### 案例 1：长 prompt 进来后首 token 延迟高

现象：prompt 长度一上去，首 token 延迟明显升高，但 decode 阶段单 token 速度还可以。

判断：
- 先看 Task2，确认 attention 是否卡在中间 score 矩阵和 HBM 读写。
- 再看 Task4，确认是否存在重复前缀、chunked prefill 是否有意义。
- 最后用 Task6 的 `TTFT / prefill_share` 验证优化收益。

常见结论：首 token 慢不等于 decode 慢，要先分清 prefill 和 decode。

### 案例 2：并发一高，整体吞吐上不去

现象：单请求还可以，但多请求同时进来后 generated tokens/s 不高，TPOT 变差。

判断：
- 先看 Task3，确认解码策略是否增加了无效计算。
- 再看 Task4，确认 prefill 和 decode 是否互相阻塞，decode batch 是否排顺。
- 最后用 Task6 比较 baseline 和 candidate 的 `throughput_gain / TTFT delta / TPOT delta`。

常见结论：吞吐问题常常不是模型本身慢，而是请求组织没有把 decode 阶段排好。

### 案例 3：cache 一边跑一边涨，batch 上不去

现象：长上下文或多轮对话下显存持续增长，batch size 被迫降低。

判断：
- 先看 Part01 `11`，理解 KV cache 随层数、头数、长度和 batch 的增长。
- 再看 `22 / 36 / 41`，确认分页、前缀复用、驱逐策略是否匹配请求分布。
- 如果仍接近显存预算，再看 Task5 的 KV cache 量化。

常见结论：cache 变大有一部分是自然增长，优化重点是减少碎片、提高复用、控制驱逐和压缩存储。

### 案例 4：量化后显存省了，但体验变差

现象：peak memory 降了，batch 能上去，但 TTFT 或 TPOT 变差。

判断：
- 先分清权重量化、KV cache 量化和 FP8 改的是哪类数据。
- 再回到 `66` 看吞吐收益是否抵消延迟退化。
- 如果在线交互场景 TTFT 退化明显，即使 throughput 更高，也不一定应该切换。

常见结论：量化不是免费收益，最终要回到 workload 和服务目标。

## 决策清单

做推理优化报告前，至少确认这些项：

- workload 是否固定：模型、backend、batch、prompt tokens、generated tokens、dtype、cache policy。
- 是否拆分 prefill 和 decode，而不是只报 total latency。
- 是否同时报告 TTFT、TPOT、throughput 和 peak memory。
- candidate 是否只改一个变量。
- 瓶颈诊断是否能解释下一步动作。
- 决策是否回到 `keep / tune / switch`，而不是只说“更快”。

## 常见误区

- 只看 throughput，不看 TTFT，导致在线交互体验退化。
- 只优化 attention kernel，不看 KV cache 和请求调度。
- 只看单条 prompt benchmark，不看请求分布。
- 把训练显存优化手段直接搬到推理场景，忽略 KV cache 是推理侧主要变量。
- 看到量化省显存就直接切换，不检查 TPOT、质量和部署复杂度。

## 相关跳转

- 完整路线见 [推理优化专题入口](./intro.md)。
- 连续链路见 [推理优化深入阅读](./walkthrough.md)。
- 需要证明慢点在哪里时，先看 [Profiling 专题](../profiling/intro.md)。
- 需要处理显存预算时，先看 [显存优化与性能调优专题](../memory_performance_tuning/intro.md)。

## 小结

推理优化不是堆技巧，而是把 workload、指标、瓶颈和候选方案放在同一张报告里判断。非项目节负责讲清机制，横向专题负责串路线，`66` 负责证明收益。
