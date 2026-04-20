"""Shared utilities for offline benchmarks."""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models.loader import load_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from somi_inference.models.base import ModelAdapter


DEFAULT_BLOCK_SIZE = 16
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-provided device string."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve a user-provided dtype string."""
    if dtype_name == "auto":
        if device.type == "cuda":
            return (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        return torch.float32

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def seed_everything(seed: int) -> None:
    """Seed RNGs used by benchmarks."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmark_adapter(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ModelAdapter, dict[str, int]]:
    """Load a model adapter for benchmarks and normalize the config."""
    adapter, raw_config = load_model(model_name)
    config_keys = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    )
    config: dict[str, int] = {}
    for key in config_keys:
        value = raw_config.get(key)
        if not isinstance(value, int):
            message = f"Expected integer config value for {key}"
            raise TypeError(message)
        config[key] = value
    config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

    model = getattr(adapter, "model", None)
    if not isinstance(model, torch.nn.Module):
        message = (
            "Benchmark requires an adapter with a torch.nn.Module `model` attribute."
        )
        raise TypeError(message)

    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(requires_grad=False)
    return adapter, config


def create_kv_manager(
    config: dict[str, int],
    *,
    num_blocks: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> KVCacheManager:
    """Create a device-aligned KV cache manager for benchmarks."""
    return KVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        n_layers=config["num_hidden_layers"],
        device=device,
        dtype=dtype,
    )


def required_blocks(tokens_per_sequence: int, block_size: int) -> int:
    """Return the number of blocks needed for a sequence length."""
    return math.ceil(tokens_per_sequence / block_size)


def synchronize(device: torch.device) -> None:
    """Synchronize the device before/after timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_runtime(
    fn: Callable[[], Any],
    *,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
    before_each: Callable[[], Any] | None = None,
) -> list[float]:
    """Measure repeated execution time of fn in seconds."""
    for _ in range(warmup_iters):
        if before_each is not None:
            before_each()
        synchronize(device)
        fn()
        synchronize(device)

    latencies: list[float] = []
    for _ in range(measure_iters):
        if before_each is not None:
            before_each()
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        latencies.append(time.perf_counter() - start)
    return latencies


def summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    """Summarize latency samples in milliseconds."""
    samples_ms = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    return {
        "mean_latency_ms": float(samples_ms.mean()),
        "median_latency_ms": float(np.percentile(samples_ms, 50)),
        "p95_latency_ms": float(np.percentile(samples_ms, 95)),
        "p99_latency_ms": float(np.percentile(samples_ms, 99)),
        "std_latency_ms": float(samples_ms.std()),
        "min_latency_ms": float(samples_ms.min()),
        "max_latency_ms": float(samples_ms.max()),
    }


def _run_git_command(*args: str) -> str | None:
    """Return git command output, or None when unavailable."""
    git_executable = shutil.which("git")
    if git_executable is None:
        return None

    try:
        completed = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_sha() -> str | None:
    """Return the current git commit SHA when available."""
    return _run_git_command("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    """Return whether tracked files differ from HEAD when available."""
    status = _run_git_command("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None
    return bool(status)


def _device_name(device: torch.device) -> str:
    """Return a human-readable device name."""
    if device.type != "cuda":
        return platform.processor() or platform.machine() or "CPU"

    if not torch.cuda.is_available():
        return "CUDA unavailable"

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_name(device_index)


def _device_capability(device: torch.device) -> str | None:
    """Return CUDA compute capability as a string when relevant."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    return f"{major}.{minor}"


def collect_environment_metadata(device: torch.device) -> dict[str, Any]:
    """Collect stable environment metadata for benchmark JSONL output."""
    device_index = device.index
    if device.type == "cuda" and device_index is None and torch.cuda.is_available():
        device_index = torch.cuda.current_device()

    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_index": device_index,
        "device_name": _device_name(device),
        "device_capability": _device_capability(device),
        "executable": sys.executable,
    }


def append_jsonl(path: str | None, payload: dict[str, Any]) -> None:
    """Append a payload to a JSONL file when requested."""
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def print_result(
    *,
    title: str,
    config: dict[str, Any],
    metrics: dict[str, float | int],
) -> None:
    """Print a compact benchmark summary."""
    print(f"\n{title:=^72}")
    for key, value in config.items():
        print(f"{key:<32} {value}")
    print("-" * 72)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<32} {value:.4f}")
        else:
            print(f"{key:<32} {value}")
