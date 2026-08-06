# 70. Distributed Inference Project | 分布式推理逻辑验证

**难度：** Hard | **环境：** CPU-first | **标签：** `项目实战`, `distributed inference`, `serving` | **目标人群：** 推理工程与系统实践

> 🚀 **云端运行环境**
>
> 本章节的实战代码可以点击以下链接在免费 GPU 算力平台上直接运行：
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/datawhalechina/llm-algo-leetcode/blob/main/02_PyTorch_Algorithms/70_Distributed_Inference_Project.ipynb)
> [![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook) *(国内推荐：魔搭社区免费实例)*


> 🚀 **云端运行环境**
>
> 本章节的实战代码优先设计为单卡、单进程可运行，适合在 Colab 或 ModelScope Notebook 中直接验证。
> 它不启动 `torch.distributed` 多进程，只用逻辑模拟的方式验证分布式推理的核心骨架。

---

## 本节导读

分布式推理项目最务实的切入点不是先搭真实多卡环境，而是先做单卡逻辑验证。这样可以在有限资源下把项目骨架、指标口径和工程决策先跑通，再决定是否迁移到真实多卡 serving。

本节把分布式推理做成一个可复用的对比项目：围绕同一批请求和同一套虚拟集群配置，记录路由结果、逻辑时延、负载均衡和通信开销，再回答“在给定约束下，这个分布式方案在逻辑上是否值得迁移到真多卡环境”。代码区只实现最小项目模板：请求路由、成本汇总、指标对比和报告生成。

本节把分布式推理拆成三个可在单卡上模拟的核心逻辑：

- **路由**：请求如何分配给不同的虚拟副本
- **切分**：tensor parallel / pipeline parallel 的逻辑开销如何进入模型
- **指标**：如何用 makespan、throughput、load balance 和 communication share 描述收益

这和 `20 FlashAttention Sim` 的思路一致：先把机制讲清楚，再谈真实环境迁移。不同的是，这一节对应的是分布式推理和 serving 骨架，而不是 attention kernel 本身。

**关键词：** `routing`, `load balance`, `communication`, `serving`

## 前置阅读

**导语：** 先看推理优化和并行策略，再进入分布式推理逻辑验证会更顺。
- [66. Inference Performance Comparison | 推理性能对比实验](./66_Inference_Performance_Comparison.md)
- [79. Distributed Parallel Benchmark | 分布式并行基准项目](./79_Distributed_Parallel_Benchmark.md)
- [2.6 核心推理优化](./2_6.md)
- [2.8 分布式并行策略](./2_8.md)
- [通信与并行专题](../topic_discussion/communication_parallel/intro.md)
- [推理优化专题](../topic_discussion/inference_optimization/intro.md)

## 相关阅读

**导语：** 如果要继续把这条线补深，可以沿着推理性能、profiling 和通信观测继续展开。
- [74. Profiling-Driven End-to-End Optimization | profiling 驱动的端到端优化](./74_Profiling_Driven_End_to_End_Optimization.md)
- [67. Quantized Inference and Deployment | 量化推理与部署](./67_Quantized_Inference_and_Deployment.md)
- [46. Communication Profiling with NCCL | NCCL 通信 profiling](./46_Communication_Profiling_with_NCCL.md)

## 本节目标

1. 用单卡、单进程模拟分布式推理的请求路由。
2. 把 tensor parallel / pipeline parallel 的逻辑代价折进请求成本。
3. 输出一份可复用的项目报告，说明在给定资源约束下，逻辑上是否值得进一步迁移到真多卡环境。

### Step 1: 定义 workload 与虚拟集群

先把“分布式推理”降成一个可以在单卡上验证的模型：

- 固定请求集合、虚拟副本数、tensor parallel / pipeline parallel 逻辑度和路由策略
- 每个请求只包含 `prompt_tokens` 和 `generate_tokens`
- 每个虚拟副本都共享同一套逻辑配置，但各自维护负载
- 请求先按 least-loaded 原则路由到某个副本，再计算对应的逻辑成本
- 成本由三部分组成：prefill、decode 和通信 / pipeline 额外开销

这样一来，Notebook 不需要真实启动多进程，也能说明分布式 serving 的核心工程问题：路由、切分和负载均衡。


```python
from typing import Dict, List


def estimate_request_cost(request: Dict[str, int], config: Dict[str, float]) -> Dict[str, float]:
    """Estimate the logical cost of one request in a distributed inference setup.

    The function stays purely local and does not require torch.distributed.
    """
    prompt_tokens = request["prompt_tokens"]
    generate_tokens = request["generate_tokens"]

    prefill_ms = prompt_tokens * config["prefill_ms_per_token"]
    decode_ms = generate_tokens * config["decode_ms_per_token"]
    comm_ms = max(config["tp_degree"] - 1, 0) * config["comm_ms_per_token"] * (
        prompt_tokens / 128.0 + generate_tokens / 64.0
    )
    pipeline_ms = max(config["pp_stages"] - 1, 0) * config["pipeline_penalty_ms"]
    total_ms = prefill_ms + decode_ms + comm_ms + pipeline_ms

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "comm_ms": comm_ms,
        "pipeline_ms": pipeline_ms,
        "total_ms": total_ms,
    }


def simulate_distributed_inference(
    requests: List[Dict[str, int]],
    num_replicas: int,
    config: Dict[str, float],
) -> Dict[str, object]:
    """Simulate routing and logical execution on a virtual distributed cluster."""
    loads = [0.0 for _ in range(num_replicas)]
    assignments: List[int] = []
    request_costs: List[Dict[str, float]] = []

    total_tokens = 0
    total_compute_ms = 0.0
    total_comm_ms = 0.0

    for request in requests:
        cost = estimate_request_cost(request, config)
        replica_idx = min(range(num_replicas), key=lambda idx: loads[idx])
        loads[replica_idx] += cost["total_ms"]

        assignments.append(replica_idx)
        request_costs.append(cost)
        total_tokens += request["prompt_tokens"] + request["generate_tokens"]
        total_compute_ms += cost["prefill_ms"] + cost["decode_ms"]
        total_comm_ms += cost["comm_ms"] + cost["pipeline_ms"]

    single_replica_time_ms = sum(loads)
    makespan_ms = max(loads) if loads else 0.0
    throughput_tok_s = (total_tokens / makespan_ms * 1000.0) if makespan_ms else 0.0
    load_gap_ms = (max(loads) - min(loads)) if loads else 0.0
    speedup_vs_single_replica = (single_replica_time_ms / makespan_ms) if makespan_ms else 0.0
    communication_share = (
        total_comm_ms / (total_compute_ms + total_comm_ms)
        if (total_compute_ms + total_comm_ms)
        else 0.0
    )

    return {
        "routing_strategy": "least_loaded",
        "num_replicas": num_replicas,
        "assignments": assignments,
        "request_costs": request_costs,
        "replica_loads_ms": loads,
        "num_requests": len(requests),
        "total_tokens": total_tokens,
        "single_replica_time_ms": single_replica_time_ms,
        "makespan_ms": makespan_ms,
        "throughput_tok_s": throughput_tok_s,
        "load_gap_ms": load_gap_ms,
        "speedup_vs_single_replica": speedup_vs_single_replica,
        "communication_share": communication_share,
    }


def format_distributed_inference_report(project_name: str, summary: Dict[str, object], recommendation: str) -> str:
    """Render a compact markdown report for the project."""
    header = "| 指标 | 数值 |"
    sep = "| --- | --- |"
    rows = [
        f"| 项目 | {project_name} |",
        f"| 路由策略 | {summary['routing_strategy']} |",
        f"| 虚拟副本数 | {summary['num_replicas']} |",
        f"| 请求数 | {summary['num_requests']} |",
        f"| total tokens | {summary['total_tokens']} |",
        f"| 单副本总时延 (ms) | {round(summary['single_replica_time_ms'], 2)} |",
        f"| makespan (ms) | {round(summary['makespan_ms'], 2)} |",
        f"| throughput (tok/s) | {round(summary['throughput_tok_s'], 2)} |",
        f"| speedup vs single replica | {round(summary['speedup_vs_single_replica'], 2)} |",
        f"| load gap (ms) | {round(summary['load_gap_ms'], 2)} |",
        f"| communication share | {round(summary['communication_share'], 3)} |",
        f"| assignment | {summary['assignments']} |",
        f"| recommendation | {recommendation} |",
    ]
    return "\n".join([header, sep] + rows)

```

### Step 2: 逻辑验证与对比

用一组最小样例验证三件事：

- 请求会按 least-loaded 的方式落到不同虚拟副本上
- 单卡模拟中，通信和 pipeline 逻辑开销会进入总时延
- 汇总指标能正确反映 makespan、throughput 和负载均衡情况

这一步的重点不是追求绝对真实，而是保证项目骨架和指标口径是稳定的。


```python
def test_distributed_inference_project():
    requests = [
        {"prompt_tokens": 128, "generate_tokens": 64},
        {"prompt_tokens": 64, "generate_tokens": 32},
    ]

    config = {
        "prefill_ms_per_token": 0.01,
        "decode_ms_per_token": 0.02,
        "comm_ms_per_token": 0.03,
        "pipeline_penalty_ms": 0.25,
        "tp_degree": 2,
        "pp_stages": 1,
    }

    summary = simulate_distributed_inference(requests, num_replicas=2, config=config)

    assert summary["routing_strategy"] == "least_loaded", "路由策略标记不正确！"
    assert summary["assignments"] == [0, 1], "请求路由没有落到不同副本上！"
    assert round(summary["single_replica_time_ms"], 2) == 3.93, "单副本总时延计算错误！"
    assert round(summary["makespan_ms"], 2) == 2.62, "makespan 计算错误！"
    assert round(summary["speedup_vs_single_replica"], 2) == 1.5, "speedup 计算错误！"
    assert round(summary["communication_share"], 3) == 0.023, "communication share 计算错误！"
    assert round(summary["throughput_tok_s"], 2) == 109923.66, "throughput 计算错误！"
    assert round(summary["load_gap_ms"], 2) == 1.31, "负载差异计算错误！"

    report = format_distributed_inference_report(
        "Distributed Inference Logic Validation",
        summary,
        "适合作为 Colab / ModelScope 上的单卡逻辑验证骨架，后续可迁移到真实多卡环境",
    )
    assert "Distributed Inference Logic Validation" in report, "报告未包含项目名称！"
    assert "least_loaded" in report, "报告未包含路由策略！"
    assert "throughput (tok/s)" in report, "报告字段不完整！"
    assert "单卡逻辑验证骨架" in report, "报告未包含推荐结论！"

    print("✅ 分布式推理逻辑验证项目通过基础校验。")
    print(report)


test_distributed_inference_project()

```

### Step 3: 复盘与迁移判断

- 先做单卡逻辑验证，再考虑真实多卡迁移。
- 先统一路由、切分和指标口径，再谈 serving 性能优化。
- 如果后续要补成真多卡项目，可以直接把这里的虚拟副本、负载和通信项映射到真实 `torch.distributed` / serving 框架中。
- 最终产物应至少包含：路由策略、成本汇总、对比报告和迁移建议。

这一页的目标是把分布式推理项目的骨架跑通，而不是替代真实部署环境。
