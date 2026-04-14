"""Profile semantic stage breakdown for one prefill pass."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import somi_inference.models.qwen2_adapter as qwen2_adapter_module
from benchmarks.common import (
    DEFAULT_BLOCK_SIZE,
    collect_environment_metadata,
    create_kv_manager,
    load_benchmark_adapter,
    required_blocks,
    resolve_device,
    resolve_dtype,
    seed_everything,
    synchronize,
)
from somi_inference.core.paged_attention import KVCacheManager
from somi_inference.models import qwen2 as qwen2_module
from somi_inference.models.qwen2 import (
    ForwardContext,
    PrefillAttentionBackend,
    QwenAttention,
    QwenMLP,
)
from somi_inference.models.qwen2_adapter import QwenAdapter
from somi_inference.tokenizer import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Iterator


DEFAULT_PROMPT = (
    "Summarize why grouped-query attention and paged KV cache matter for "
    "single-request LLM inference. Explain the main memory bottlenecks, how "
    "prefill differs from decode, and why a Triton fast path can improve TTFT "
    "even when the end-to-end gain is smaller than the microbenchmark gain."
)
STAGE_PREFIX = "prefill/"
PRIMARY_STAGE_LABELS = (
    "project_qkv",
    "write_kv",
    "mlp",
    "lm_head",
)


@dataclass
class StageTiming:
    """One semantic stage timing summary."""

    label: str
    time_ms: float
    share_pct: float


@dataclass
class ProfileResult:
    """Profile summary for one backend on one prompt."""

    requested_backend: PrefillAttentionBackend
    resolved_attention_backend: str
    prompt_len: int
    measurement_name: str
    total_time_ms: float
    stage_timings: list[StageTiming]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Profile semantic prefill stages with torch.profiler."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model / cache dtype. 'auto' picks a device-appropriate default.",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        nargs="+",
        default=["torch_ref", "triton"],
        choices=["auto", "torch_ref", "triton"],
        help="Prefill attention backends to profile, run serially in order.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="KV cache block size.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of untimed warmup prefills before profiling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic prompt generation.",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=DEFAULT_PROMPT,
        help="Text prompt to profile when --synthetic-prompt-len is omitted.",
    )
    parser.add_argument(
        "--synthetic-prompt-len",
        type=int,
        default=None,
        help="Use a random synthetic prompt with this many tokens instead.",
    )
    return parser.parse_args()


def _build_input_ids(
    args: argparse.Namespace,
    *,
    model_name: str,
    config: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    if args.synthetic_prompt_len is not None:
        prompt_len = args.synthetic_prompt_len
        if prompt_len <= 0:
            message = "--synthetic-prompt-len must be positive."
            raise ValueError(message)
        input_ids = torch.randint(
            config["vocab_size"],
            (1, prompt_len),
            device=device,
            dtype=torch.long,
        )
        return input_ids, f"synthetic:{prompt_len}"

    tokenizer = Tokenizer(model_name)
    token_ids = tokenizer.encode(args.prompt_text)
    if not token_ids:
        message = "Prompt text produced zero tokens; provide a non-empty prompt."
        raise ValueError(message)
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    return input_ids, "text"


def _resolve_attention_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: PrefillAttentionBackend,
) -> str:
    if backend == "auto":
        return (
            "triton"
            if qwen2_module.triton_causal_attention_supported(q, k, v)
            else "torch_ref"
        )
    return backend


def _measurement_name(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _time_attr(device: torch.device) -> str:
    if device.type == "cuda":
        return "device_time_total"
    return "cpu_time_total"


def _event_time_ms(event: object | None, attr_name: str) -> float:
    if event is None:
        return 0.0
    value = getattr(event, attr_name, 0.0)
    return float(value) / 1000.0


@contextmanager
def _install_stage_hooks() -> Iterator[None]:
    original_attention_forward = QwenAttention.forward
    original_project_qkv = QwenAttention._project_qkv  # noqa: SLF001
    original_mlp_forward = QwenMLP.forward
    original_lm_head = QwenAdapter._lm_head  # noqa: SLF001
    original_write_kv = KVCacheManager.write_kv
    original_causal_attention = qwen2_adapter_module.causal_attention

    def wrapped_attention_forward(
        self: QwenAttention,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        with record_function(f"{STAGE_PREFIX}attn_block"):
            return original_attention_forward(self, hidden_states, cos, sin, ctx)

    def wrapped_project_qkv(
        self: QwenAttention,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with record_function(f"{STAGE_PREFIX}project_qkv"):
            return original_project_qkv(self, hidden_states, cos, sin)

    def wrapped_mlp_forward(self: QwenMLP, x: torch.Tensor) -> torch.Tensor:
        with record_function(f"{STAGE_PREFIX}mlp"):
            return original_mlp_forward(self, x)

    def wrapped_lm_head(
        self: QwenAdapter, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        with record_function(f"{STAGE_PREFIX}lm_head"):
            return original_lm_head(self, hidden_states)

    def wrapped_write_kv(
        self: KVCacheManager,
        seq_id: int,
        layer_idx: int,
        layer_key: torch.Tensor,
        layer_value: torch.Tensor,
    ) -> None:
        with record_function(f"{STAGE_PREFIX}write_kv"):
            original_write_kv(self, seq_id, layer_idx, layer_key, layer_value)

    def wrapped_causal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        backend: PrefillAttentionBackend = "auto",
    ) -> torch.Tensor:
        resolved_backend = _resolve_attention_backend(q, k, v, backend)
        with record_function(f"{STAGE_PREFIX}causal_attention[{resolved_backend}]"):
            return original_causal_attention(q, k, v, backend=backend)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(QwenAttention, "forward", wrapped_attention_forward)
        )
        stack.enter_context(
            patch.object(QwenAttention, "_project_qkv", wrapped_project_qkv)
        )
        stack.enter_context(patch.object(QwenMLP, "forward", wrapped_mlp_forward))
        stack.enter_context(patch.object(QwenAdapter, "_lm_head", wrapped_lm_head))
        stack.enter_context(patch.object(KVCacheManager, "write_kv", wrapped_write_kv))
        stack.enter_context(
            patch.object(
                qwen2_adapter_module,
                "causal_attention",
                wrapped_causal_attention,
            )
        )
        yield


def _run_prefill_once(
    adapter: QwenAdapter,
    kv_manager: KVCacheManager,
    input_ids: torch.Tensor,
    seq_id: int,
) -> torch.Tensor:
    kv_manager.register_sequence(seq_id)
    try:
        with torch.inference_mode():
            return adapter.prefill(input_ids, kv_manager, seq_id)
    finally:
        kv_manager.free_sequence(seq_id)


def _summarize_profile(
    profiler: torch.profiler.profile,
    *,
    requested_backend: PrefillAttentionBackend,
    prompt_len: int,
    device: torch.device,
) -> ProfileResult:
    time_attr = _time_attr(device)
    by_key = {event.key: event for event in profiler.key_averages()}
    total_time_ms = _event_time_ms(by_key.get(f"{STAGE_PREFIX}total"), time_attr)

    attention_labels = sorted(
        key
        for key in by_key
        if key.startswith(f"{STAGE_PREFIX}causal_attention[")
    )
    if attention_labels:
        attention_label = attention_labels[0]
        resolved_backend = attention_label.removeprefix(
            f"{STAGE_PREFIX}causal_attention["
        ).removesuffix("]")
    else:
        attention_label = None
        resolved_backend = "unknown"

    stage_timings: list[StageTiming] = []
    tracked_time_ms = 0.0

    for label in PRIMARY_STAGE_LABELS:
        stage_label = f"{STAGE_PREFIX}{label}"
        stage_time_ms = _event_time_ms(by_key.get(stage_label), time_attr)
        tracked_time_ms += stage_time_ms
        share_pct = (stage_time_ms / total_time_ms * 100.0) if total_time_ms else 0.0
        stage_timings.append(
            StageTiming(
                label=label,
                time_ms=stage_time_ms,
                share_pct=share_pct,
            )
        )

    if attention_label is not None:
        attention_time_ms = _event_time_ms(by_key.get(attention_label), time_attr)
        tracked_time_ms += attention_time_ms
        share_pct = (
            attention_time_ms / total_time_ms * 100.0 if total_time_ms else 0.0
        )
        stage_timings.insert(
            1,
            StageTiming(
                label=f"causal_attention[{resolved_backend}]",
                time_ms=attention_time_ms,
                share_pct=share_pct,
            ),
        )

    other_time_ms = max(total_time_ms - tracked_time_ms, 0.0)
    other_share_pct = (other_time_ms / total_time_ms * 100.0) if total_time_ms else 0.0
    stage_timings.append(
        StageTiming(
            label="other",
            time_ms=other_time_ms,
            share_pct=other_share_pct,
        )
    )

    return ProfileResult(
        requested_backend=requested_backend,
        resolved_attention_backend=resolved_backend,
        prompt_len=prompt_len,
        measurement_name=_measurement_name(device),
        total_time_ms=total_time_ms,
        stage_timings=stage_timings,
    )


def _profile_backend(
    adapter: QwenAdapter,
    config: dict[str, int],
    input_ids: torch.Tensor,
    *,
    backend: PrefillAttentionBackend,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_iters: int,
) -> ProfileResult:
    num_blocks = required_blocks(input_ids.size(1), block_size) + 1
    kv_manager = create_kv_manager(
        config,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    adapter.prefill_attention_backend = backend
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with _install_stage_hooks():
        for seq_id in range(warmup_iters):
            synchronize(device)
            _run_prefill_once(adapter, kv_manager, input_ids, seq_id)
            synchronize(device)

        seq_id = warmup_iters
        synchronize(device)
        with (
            profile(activities=activities) as profiler,
            record_function(f"{STAGE_PREFIX}total"),
        ):
            _run_prefill_once(adapter, kv_manager, input_ids, seq_id)
        synchronize(device)

    return _summarize_profile(
        profiler,
        requested_backend=backend,
        prompt_len=input_ids.size(1),
        device=device,
    )


def _print_header(
    *,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    prompt_source: str,
    prompt_len: int,
    environment: dict[str, object],
) -> None:
    print("=== Prefill Profile ===")
    print(f"model={model_name}")
    print(f"device={device} ({environment['device_name']})")
    print(f"dtype={dtype}")
    print(f"prompt_source={prompt_source}")
    print(f"prompt_len={prompt_len}")
    print(f"torch={environment['torch_version']}")


def _print_result(result: ProfileResult) -> None:
    print(
        f"\n--- backend={result.requested_backend}"
        f" (resolved_attention={result.resolved_attention_backend}) ---"
    )
    print(f"total_{result.measurement_name}_ms={result.total_time_ms:.3f}")
    for stage in result.stage_timings:
        print(
            f"{stage.label:28} "
            f"{stage.time_ms:9.3f} ms "
            f"{stage.share_pct:6.2f}%"
        )


def main() -> None:
    """Run the prefill profiler."""
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    seed_everything(args.seed)

    adapter, config = load_benchmark_adapter(args.model_name, device, dtype)
    if not isinstance(adapter, QwenAdapter):
        message = "Prefill profiler currently expects a QwenAdapter."
        raise TypeError(message)

    input_ids, prompt_source = _build_input_ids(
        args,
        model_name=args.model_name,
        config=config,
        device=device,
    )
    if input_ids.size(1) > config["max_position_embeddings"]:
        message = (
            f"prompt_len={input_ids.size(1)} exceeds "
            f"max_position_embeddings={config['max_position_embeddings']}"
        )
        raise ValueError(message)

    environment = collect_environment_metadata(device)
    _print_header(
        model_name=args.model_name,
        device=device,
        dtype=dtype,
        prompt_source=prompt_source,
        prompt_len=input_ids.size(1),
        environment=environment,
    )

    for backend_name in args.attention_backends:
        backend = cast("PrefillAttentionBackend", backend_name)
        result = _profile_backend(
            adapter,
            config,
            input_ids,
            backend=backend,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
