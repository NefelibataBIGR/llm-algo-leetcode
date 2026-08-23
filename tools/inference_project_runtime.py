"""Notebook helpers shared by inference projects 66--70.

The helper keeps the optional GPU path explicit: a notebook can stay runnable
without vLLM, while a GPU learner gets model resolution, dtype selection, free
port selection, service cleanup and result persistence from one place.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def locate_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook working directory."""

    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "tools").is_dir() and (candidate / "02_PyTorch_Algorithms").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError("未找到项目根目录，请从仓库启动 Notebook 或先 clone 仓库。")


def shared_project_config(
    *,
    model: str,
    backend: str = "local",
    dtype: str = "unknown",
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    batch: int = 1,
    concurrency: int = 1,
    cache_policy: str = "default",
    **extra: Any,
) -> dict[str, Any]:
    """Build the common 66--70 configuration block."""

    config = {
        "model": model,
        "backend": backend,
        "dtype": dtype,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "batch": batch,
        "concurrency": concurrency,
        "cache_policy": cache_policy,
    }
    errors = validate_backend_config(config)
    if errors:
        raise ValueError("invalid backend config: " + "; ".join(errors))
    config.update({key: value for key, value in extra.items() if value is not None})
    return config


def validate_backend_config(config: Mapping[str, Any]) -> list[str]:
    """Validate common 66--70 fields without starting a server."""

    errors: list[str] = []
    for key in ("model", "backend", "dtype", "cache_policy"):
        if not str(config.get(key, "")).strip():
            errors.append(f"missing config: {key}")
    for key in ("batch", "concurrency"):
        if key in config and int(config[key]) <= 0:
            errors.append(f"invalid config: {key}")
    for key in ("prompt_tokens", "generated_tokens"):
        if config.get(key) is not None and int(config[key]) <= 0:
            errors.append(f"invalid config: {key}")
    return errors


def runtime_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Capture the environment used by a backend run."""

    snapshot: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    if torch_module is None:
        return snapshot
    snapshot["torch"] = getattr(torch_module, "__version__", None)
    snapshot["torch_cuda"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        snapshot["device"] = cuda.get_device_name(0)
    else:
        snapshot["device"] = "cpu"
    return snapshot


def save_project_result(
    path: str | Path,
    *,
    project: str,
    strategy: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    quality: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    strategy_metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save one project result using the shared schema."""

    from inference_result_schema import make_result, save_result

    result = make_result(
        project=project,
        strategy=strategy,
        config=config,
        metrics=metrics,
        quality=quality,
        decision=decision,
        strategy_metrics=strategy_metrics,
    )
    output = save_result(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return output


def start_optional_vllm(
    *,
    model_id: str,
    model_source: str = "auto",
    cache_dir: str | Path = "model_cache",
    dtype: str = "auto",
    vllm_command: str | None = None,
    vllm_environment: str | None = None,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = True,
    served_model_name: str | None = None,
) -> tuple[Any, Path, int, str, str]:
    """Resolve a model and launch vLLM for an optional Practice-P2 run.

    Returns ``(server, log_path, port, selected_dtype, model_path)``. The
    caller owns the server and must call ``stop_optional_vllm`` in ``finally``.
    """

    root = locate_repo_root()
    from tools.backend_runtime import resolve_model, start_vllm

    model_path = resolve_model(model_id, model_source, cache_dir=cache_dir)
    server, log_path, port, selected_dtype = start_vllm(
        model_path,
        dtype,
        vllm_command=vllm_command,
        vllm_environment=vllm_environment,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        served_model_name=served_model_name or model_id,
    )
    return server, log_path, port, selected_dtype, str(model_path)


def stop_optional_vllm(server: Any, log_path: str | Path) -> None:
    """Stop a server started by :func:`start_optional_vllm`."""

    from tools.backend_runtime import stop_backend

    stop_backend(server, Path(log_path))


def run_backend_benchmark(
    *,
    project: str,
    base_url: str,
    model: str,
    label: str,
    output: str | Path,
    workload: str = "benchmarks/workloads/fixed.jsonl",
    num_prompts: int = 5,
    max_tokens: int = 64,
    concurrency: int = 1,
    warmup: int = 1,
    backend: str = "vllm",
    dtype: str = "unknown",
    batch: int = 1,
    cache_policy: str = "default",
) -> dict[str, Any]:
    """Run the common OpenAI-compatible benchmark and load its JSON output."""

    root = locate_repo_root()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    command = [
        sys.executable,
        str(root / "tools/benchmark_inference_backend.py"),
        "--base-url", base_url,
        "--model", model,
        "--label", label,
        "--project", project,
        "--backend", backend,
        "--dtype", dtype,
        "--batch", str(batch),
        "--cache-policy", cache_policy,
        "--workload", str(root / workload),
        "--num-prompts", str(num_prompts),
        "--max-tokens", str(max_tokens),
        "--concurrency", str(concurrency),
        "--warmup", str(warmup),
        "--output", str(output_path),
    ]
    subprocess.run(command, cwd=root, check=True)
    return json.loads(output_path.read_text(encoding="utf-8"))
