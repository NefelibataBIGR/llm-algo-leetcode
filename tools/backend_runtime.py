"""Notebook helpers for starting and stopping an optional local inference backend."""

from __future__ import annotations

import socket
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def resolve_vllm_command(command: str | None = None, environment: str | None = None) -> str:
    """Resolve a vLLM executable in the current or a sibling conda environment."""
    if command:
        candidate = Path(command).expanduser()
        if candidate.is_file():
            return str(candidate)
        if Path(command).name == command:
            found = shutil.which(command)
            if found:
                return found
        raise FileNotFoundError(f"找不到 vLLM 可执行文件: {command}")

    if environment:
        conda_envs = [Path(sys.prefix).parent, Path(sys.prefix).parent.parent / "envs"]
        for env_root in conda_envs:
            candidate = env_root / environment / "bin" / "vllm"
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(
            f"找不到 conda 环境 {environment!r} 中的 vllm，请确认环境已安装。"
        )

    found = shutil.which("vllm")
    if found:
        return found
    raise FileNotFoundError("当前环境中找不到 vllm，请设置 VLLM_ENV 或 VLLM_COMMAND。")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def choose_dtype(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    except ImportError:
        return "float16"


def resolve_model(model_id: str, source: str = "huggingface", cache_dir: str = "./model_cache") -> str:
    from .model_runtime import resolve_model as resolve_shared_model
    return resolve_shared_model(model_id, source=source, cache_dir=cache_dir)


def wait_until_ready(port: int, timeout_s: float = 120.0) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # backend may need time to load weights
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Backend did not become ready at {url}: {last_error}")


def start_vllm(model_path: str, dtype: str = "auto", port: int | None = None,
               log_path: str = "benchmarks/results/66_backend.log",
               vllm_command: str | None = None, vllm_environment: str | None = None,
               max_model_len: int = 2048,
               gpu_memory_utilization: float = 0.8,
               enforce_eager: bool = False, served_model_name: str | None = None):
    port = port or find_free_port()
    dtype = choose_dtype(dtype)
    vllm_command = resolve_vllm_command(vllm_command, vllm_environment)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    command = [
        vllm_command, "serve", model_path,
        "--dtype", dtype,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    if served_model_name:
        command.extend(["--served-model-name", served_model_name])
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        wait_until_ready(port)
    except Exception:
        stop_backend(process, log_file)
        raise
    return process, log_file, port, dtype


def stop_backend(process, log_file=None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if log_file is not None:
        log_file.close()
