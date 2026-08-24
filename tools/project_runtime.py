"""Low-risk runtime helpers shared by real training project notebooks.

These helpers validate configuration and record environment metadata. They do
not choose hyperparameters, retry OOMs, or change an experiment's workload.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Mapping


def ensure_output_path(root: str | Path, relative_path: str | Path) -> Path:
    """Create the parent directory and return an absolute result path."""

    output = Path(relative_path)
    if not output.is_absolute():
        output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Fail early on invalid shared training benchmark parameters."""

    positive = ("batch_size", "seq_len", "warmup", "iters")
    for key in positive:
        value = config.get(key)
        if not isinstance(value, int) or value < 0 or (key in {"batch_size", "seq_len", "iters"} and value == 0):
            raise ValueError(f"{key} 必须是有效的非负整数，当前为 {value!r}")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed 必须是整数")
    if config.get("learning_rate") is not None and float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate 必须大于 0")


def runtime_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    """Return reproducibility metadata without requiring CUDA."""

    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    if torch_module is None:
        return snapshot
    snapshot["torch"] = getattr(torch_module, "__version__", None)
    snapshot["torch_cuda"] = getattr(torch_module.version, "cuda", None)
    available = bool(torch_module.cuda.is_available())
    snapshot["cuda_available"] = available
    if available:
        snapshot["device"] = torch_module.cuda.get_device_name(0)
        snapshot["device_capability"] = list(torch_module.cuda.get_device_capability(0))
    return snapshot


def require_input_file(path: str | Path, label: str) -> Path:
    """Validate a required upstream report before a downstream project runs."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{label} 不存在：{input_path}。请先完成上游项目并保存 JSON。")
    if input_path.suffix.lower() != ".json":
        raise ValueError(f"{label} 必须是 JSON 文件：{input_path}")
    return input_path
