"""Render README-ready resume benchmark summaries and charts."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

README_START = "<!-- resume-benchmarks:start -->"
README_END = "<!-- resume-benchmarks:end -->"

VARIANT_DISPLAY = {
    "baseline": "Baseline",
    "prefill_bundle": "Triton Prefill + MLP",
    "decode_bundle": "Triton Decode",
    "full_triton": "Full Triton",
}
VARIANT_ORDER = ("baseline", "prefill_bundle", "decode_bundle", "full_triton")


@dataclass(frozen=True)
class PrefillSummaryRow:
    """One prefill comparison row."""

    prompt_len: int
    baseline_ms: float
    triton_ms: float
    speedup: float


@dataclass(frozen=True)
class PagedAttentionSummaryRow:
    """One paged-attention comparison row."""

    batch_size: int
    seq_len: int
    baseline_ms: float
    triton_ms: float
    speedup: float


@dataclass(frozen=True)
class E2ESummary:
    """Aggregated deterministic e2e session metrics for one variant."""

    variant: str
    mean_ttft_ms: float
    mean_turn_latency_ms: float
    session_total_time_ms: float
    total_tokens_per_s: float


@dataclass(frozen=True)
class EngineSummary:
    """Aggregated engine metrics for one variant."""

    variant: str
    request_throughput: float
    total_tokens_per_s: float
    duration_s: float


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Render charts and markdown from resume benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing the JSONL outputs from run_resume_benchmarks.sh.",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        required=True,
        help="Directory where committed SVG charts should be written.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        required=True,
        help="README file whose generated benchmark block should be refreshed.",
    )
    parser.add_argument(
        "--local-notes",
        type=Path,
        required=True,
        help="Gitignored markdown file for resume bullets and metric notes.",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load JSONL rows grouped by benchmark name and variant."""
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        benchmark_name, variant = path.stem.split("__", maxsplit=1)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results.setdefault(benchmark_name, {})[variant] = rows
    return results


def require_variant_rows(
    results: dict[str, dict[str, list[dict[str, Any]]]],
    benchmark_name: str,
    variant: str,
) -> list[dict[str, Any]]:
    """Return rows for one benchmark/variant pair, or fail loudly."""
    benchmark_rows = results.get(benchmark_name)
    if benchmark_rows is None or variant not in benchmark_rows:
        message = f"Missing results for {benchmark_name} / {variant}."
        raise ValueError(message)
    return benchmark_rows[variant]


def summarize_prefill(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[PrefillSummaryRow]:
    """Summarize baseline vs Triton prefill latency."""
    baseline_rows = require_variant_rows(results, "prefill", "baseline")
    triton_rows = require_variant_rows(results, "prefill", "prefill_bundle")
    baseline_by_prompt = {row["config"]["prompt_len"]: row for row in baseline_rows}
    triton_by_prompt = {row["config"]["prompt_len"]: row for row in triton_rows}
    prompt_lens = sorted(set(baseline_by_prompt) & set(triton_by_prompt))
    return [
        PrefillSummaryRow(
            prompt_len=prompt_len,
            baseline_ms=baseline_by_prompt[prompt_len]["metrics"]["mean_latency_ms"],
            triton_ms=triton_by_prompt[prompt_len]["metrics"]["mean_latency_ms"],
            speedup=(
                baseline_by_prompt[prompt_len]["metrics"]["mean_latency_ms"]
                / triton_by_prompt[prompt_len]["metrics"]["mean_latency_ms"]
            ),
        )
        for prompt_len in prompt_lens
    ]


def summarize_paged_attention(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[PagedAttentionSummaryRow]:
    """Summarize baseline vs Triton paged-attention latency."""
    baseline_rows = require_variant_rows(results, "paged_attention", "baseline")
    triton_rows = require_variant_rows(results, "paged_attention", "decode_bundle")
    baseline_by_key = {
        (row["config"]["batch_size"], row["config"]["seq_len"]): row
        for row in baseline_rows
    }
    triton_by_key = {
        (row["config"]["batch_size"], row["config"]["seq_len"]): row for row in triton_rows
    }
    keys = sorted(set(baseline_by_key) & set(triton_by_key))
    return [
        PagedAttentionSummaryRow(
            batch_size=batch_size,
            seq_len=seq_len,
            baseline_ms=baseline_by_key[(batch_size, seq_len)]["metrics"][
                "mean_latency_ms"
            ],
            triton_ms=triton_by_key[(batch_size, seq_len)]["metrics"][
                "mean_latency_ms"
            ],
            speedup=(
                baseline_by_key[(batch_size, seq_len)]["metrics"]["mean_latency_ms"]
                / triton_by_key[(batch_size, seq_len)]["metrics"]["mean_latency_ms"]
            ),
        )
        for batch_size, seq_len in keys
    ]


def summarize_e2e_variant(rows: list[dict[str, Any]], variant: str) -> E2ESummary:
    """Aggregate deterministic e2e turn rows into one session summary."""
    total_time_s = 0.0
    total_input_tokens = 0.0
    total_output_tokens = 0.0
    ttft_values: list[float] = []
    turn_latency_values: list[float] = []
    session_total_time_ms = 0.0

    for row in rows:
        config = row["config"]
        metrics = row["metrics"]
        measure_iters = float(config["measure_iters"])
        mean_latency_ms = float(metrics["mean_latency_ms"])
        total_time_s += mean_latency_ms * measure_iters / 1000.0
        total_input_tokens += float(config["prompt_tokens"]) * measure_iters
        total_output_tokens += float(metrics["mean_output_tokens"]) * measure_iters
        ttft_values.append(float(metrics["ttft_mean_latency_ms"]))
        turn_latency_values.append(mean_latency_ms)
        session_total_time_ms += mean_latency_ms

    return E2ESummary(
        variant=variant,
        mean_ttft_ms=sum(ttft_values) / len(ttft_values),
        mean_turn_latency_ms=sum(turn_latency_values) / len(turn_latency_values),
        session_total_time_ms=session_total_time_ms,
        total_tokens_per_s=(total_input_tokens + total_output_tokens) / total_time_s,
    )


def summarize_e2e(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[E2ESummary]:
    """Aggregate all e2e variants."""
    summaries: list[E2ESummary] = []
    for variant in VARIANT_ORDER:
        rows = require_variant_rows(results, "e2e", variant)
        summaries.append(summarize_e2e_variant(rows, variant))
    return summaries


def summarize_engine(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[EngineSummary]:
    """Aggregate all engine variants."""
    summaries: list[EngineSummary] = []
    for variant in VARIANT_ORDER:
        row = require_variant_rows(results, "engine", variant)[0]
        metrics = row["metrics"]
        summaries.append(
            EngineSummary(
                variant=variant,
                request_throughput=float(metrics["request_throughput"]),
                total_tokens_per_s=float(metrics["total_tokens_per_s"]),
                duration_s=float(metrics["duration_s"]),
            )
        )
    return summaries


def collect_environment(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Return the environment metadata from the first available row."""
    for benchmark_rows in results.values():
        for rows in benchmark_rows.values():
            if rows:
                return rows[0]["environment"]
    message = "No benchmark rows were found."
    raise ValueError(message)


def collect_run_config(
    results: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Return a representative config payload from the generated run."""
    for benchmark_name in ("engine", "e2e", "prefill", "paged_attention"):
        benchmark_rows = results.get(benchmark_name)
        if benchmark_rows is None:
            continue
        for variant in VARIANT_ORDER:
            rows = benchmark_rows.get(variant)
            if rows:
                return rows[0]["config"]
        for rows in benchmark_rows.values():
            if rows:
                return rows[0]["config"]
    message = "No benchmark config rows were found."
    raise ValueError(message)


def update_readme(readme_path: Path, replacement_block: str) -> None:
    """Replace the generated benchmark block inside README."""
    content = readme_path.read_text(encoding="utf-8")
    start_idx = content.find(README_START)
    end_idx = content.find(README_END)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        message = "README is missing resume benchmark markers."
        raise ValueError(message)

    start_idx += len(README_START)
    updated_content = (
        content[:start_idx]
        + "\n"
        + replacement_block.strip()
        + "\n"
        + content[end_idx:]
    )
    readme_path.write_text(updated_content, encoding="utf-8")


def svg_grouped_bar_chart(
    *,
    path: Path,
    title: str,
    y_label: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
) -> None:
    """Render a compact grouped bar chart as SVG."""
    width = 920
    height = 460
    margin_left = 72
    margin_right = 24
    margin_top = 56
    margin_bottom = 84
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(max(values) for _, values, _ in series)
    max_value = 1.0 if max_value <= 0 else max_value * 1.15
    y_ticks = 5
    group_width = plot_width / max(len(categories), 1)
    bar_gap = 10
    bars_per_group = len(series)
    total_bar_gap = bar_gap * (bars_per_group - 1)
    bar_width = min(44.0, (group_width - 24 - total_bar_gap) / max(bars_per_group, 1))

    def y_pos(value: float) -> float:
        return margin_top + plot_height - (value / max_value) * plot_height

    elements: list[str] = [
        (
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            'font-size="20" font-weight="600" fill="#111827">'
            f"{html.escape(title)}</text>"
        ),
        (
            f'<text x="20" y="{margin_top + plot_height / 2:.1f}" '
            'text-anchor="middle" font-size="13" fill="#4b5563" '
            f'transform="rotate(-90 20 {margin_top + plot_height / 2:.1f})">'
            f"{html.escape(y_label)}</text>"
        ),
    ]

    for tick_idx in range(y_ticks + 1):
        tick_value = max_value * tick_idx / y_ticks
        y = y_pos(tick_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" '
            f'x2="{width - margin_right}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            'font-size="12" fill="#6b7280">'
            f"{tick_value:.1f}</text>"
        )

    for group_idx, category in enumerate(categories):
        group_x = margin_left + group_idx * group_width
        total_bars_width = bars_per_group * bar_width + total_bar_gap
        start_x = group_x + (group_width - total_bars_width) / 2
        for series_idx, (_, values, color) in enumerate(series):
            value = values[group_idx]
            bar_height = (value / max_value) * plot_height
            x = start_x + series_idx * (bar_width + bar_gap)
            y = margin_top + plot_height - bar_height
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="4" fill="{color}"/>'
            )
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                'font-size="11" fill="#111827">'
                f"{value:.2f}</text>"
            )
        elements.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{height - 28}" text-anchor="middle" '
            'font-size="12" fill="#374151">'
            f"{html.escape(category)}</text>"
        )

    legend_x = margin_left
    legend_y = height - 56
    for label, _, color in series:
        elements.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="14" height="14" '
            f'rx="3" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{legend_x + 22}" y="{legend_y + 12}" font-size="12" fill="#374151">'
            f"{html.escape(label)}</text>"
        )
        legend_x += 22 + max(90, len(label) * 7)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 920 460" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
        + "".join(elements)
        + "</svg>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def render_prefill_table(rows: list[PrefillSummaryRow]) -> str:
    """Render the prefill markdown table."""
    lines = [
        "| Prompt Tokens | Baseline Mean | Triton Mean | Speedup |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.prompt_len} | {row.baseline_ms:.2f} ms | "
            f"{row.triton_ms:.2f} ms | {row.speedup:.2f}x |"
        )
    return "\n".join(lines)


def render_paged_attention_table(rows: list[PagedAttentionSummaryRow]) -> str:
    """Render the paged-attention markdown table."""
    lines = [
        "| Batch | Seq Len | Baseline Mean | Triton Mean | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.batch_size} | {row.seq_len} | {row.baseline_ms:.3f} ms | "
            f"{row.triton_ms:.3f} ms | {row.speedup:.2f}x |"
        )
    return "\n".join(lines)


def render_e2e_table(rows: list[E2ESummary]) -> str:
    """Render the e2e markdown table."""
    baseline = rows[0]
    lines = [
        "| Variant | Mean TTFT | Mean Turn Latency | Session Time | Total Tok/s | Δ TTFT vs Baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ttft_delta = (baseline.mean_ttft_ms - row.mean_ttft_ms) / baseline.mean_ttft_ms
        lines.append(
            f"| {VARIANT_DISPLAY[row.variant]} | {row.mean_ttft_ms:.2f} ms | "
            f"{row.mean_turn_latency_ms:.2f} ms | {row.session_total_time_ms:.2f} ms | "
            f"{row.total_tokens_per_s:.2f} | {ttft_delta * 100:+.1f}% |"
        )
    return "\n".join(lines)


def render_engine_table(rows: list[EngineSummary]) -> str:
    """Render the engine markdown table."""
    baseline = rows[0]
    lines = [
        "| Variant | Request/s | Total Tok/s | Run Duration | Δ Tok/s vs Baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        tok_delta = (
            row.total_tokens_per_s - baseline.total_tokens_per_s
        ) / baseline.total_tokens_per_s
        lines.append(
            f"| {VARIANT_DISPLAY[row.variant]} | {row.request_throughput:.3f} | "
            f"{row.total_tokens_per_s:.2f} | {row.duration_s:.2f} s | {tok_delta * 100:+.1f}% |"
        )
    return "\n".join(lines)


def render_key_takeaways(
    prefill_rows: list[PrefillSummaryRow],
    paged_rows: list[PagedAttentionSummaryRow],
    e2e_rows: list[E2ESummary],
    engine_rows: list[EngineSummary],
) -> list[str]:
    """Build the short summary bullets for README and local notes."""
    longest_prefill = max(prefill_rows, key=lambda row: row.prompt_len)
    fastest_decode = max(paged_rows, key=lambda row: row.speedup)
    baseline_e2e = e2e_rows[0]
    best_e2e = min(e2e_rows[1:], key=lambda row: row.mean_ttft_ms)
    baseline_engine = engine_rows[0]
    best_engine = max(engine_rows[1:], key=lambda row: row.total_tokens_per_s)
    engine_ratio = best_engine.total_tokens_per_s / baseline_engine.total_tokens_per_s
    engine_prefill_variants = [
        row for row in engine_rows if row.variant in {"prefill_bundle", "full_triton"}
    ]
    worst_engine_ratio = min(
        row.total_tokens_per_s / baseline_engine.total_tokens_per_s
        for row in engine_prefill_variants
    )

    if engine_ratio > 1.02:
        engine_takeaway = (
            f"{VARIANT_DISPLAY[best_engine.variant]} raises engine total throughput "
            f"from {baseline_engine.total_tokens_per_s:.2f} tok/s to "
            f"{best_engine.total_tokens_per_s:.2f} tok/s."
        )
    elif engine_ratio >= 0.98:
        engine_takeaway = (
            f"Engine throughput stays roughly flat with {VARIANT_DISPLAY[best_engine.variant]} "
            f"({baseline_engine.total_tokens_per_s:.2f} tok/s → "
            f"{best_engine.total_tokens_per_s:.2f} tok/s)."
        )
    else:
        engine_takeaway = (
            f"Engine throughput currently regresses on Triton-enabled variants; the best "
            f"non-baseline result is {best_engine.total_tokens_per_s:.2f} tok/s versus "
            f"{baseline_engine.total_tokens_per_s:.2f} tok/s."
        )

    return [
        (
            f"Triton prefill + MLP cuts {longest_prefill.prompt_len}-token prefill "
            f"latency from {longest_prefill.baseline_ms:.2f} ms to "
            f"{longest_prefill.triton_ms:.2f} ms ({longest_prefill.speedup:.2f}x)."
        ),
        (
            f"Triton paged attention is fastest on batch {fastest_decode.batch_size} / "
            f"seq {fastest_decode.seq_len}, improving raw decode-kernel latency by "
            f"{fastest_decode.speedup:.2f}x."
        ),
        (
            f"{VARIANT_DISPLAY[best_e2e.variant]} lowers agent-session mean TTFT "
            f"from {baseline_e2e.mean_ttft_ms:.2f} ms to {best_e2e.mean_ttft_ms:.2f} ms."
        ),
        engine_takeaway,
        (
            f"Triton-prefill engine variants bottom out at {worst_engine_ratio * 100:.1f}% "
            "of baseline throughput on the current scheduler path."
        ),
    ]


def render_readme_block(
    *,
    results_dir: Path,
    environment: dict[str, Any],
    run_config: dict[str, Any],
    prefill_rows: list[PrefillSummaryRow],
    paged_rows: list[PagedAttentionSummaryRow],
    e2e_rows: list[E2ESummary],
    engine_rows: list[EngineSummary],
) -> str:
    """Render the markdown block that lives inside the README markers."""
    takeaways = render_key_takeaways(prefill_rows, paged_rows, e2e_rows, engine_rows)
    run_name = results_dir.name
    device_name = environment["device_name"]
    python_version = environment["python_version"]
    torch_version = environment["torch_version"]
    cuda_version = environment["cuda_version"]
    model_name = str(run_config["model_name"])
    dtype = str(run_config["dtype"]).removeprefix("torch.")
    workload = str(run_config["workload"])
    preset = str(run_config["preset"])
    if "min_output_tokens" in run_config and "max_output_tokens" in run_config:
        min_output_tokens = int(run_config["min_output_tokens"])
        max_output_tokens = int(run_config["max_output_tokens"])
    else:
        requested_output_tokens = int(
            run_config.get("requested_output_tokens", run_config["max_new_tokens"])
        )
        min_output_tokens = requested_output_tokens
        max_output_tokens = requested_output_tokens
    if min_output_tokens == max_output_tokens:
        output_tokens_label = str(min_output_tokens)
    else:
        output_tokens_label = f"{min_output_tokens}..{max_output_tokens}"

    return "\n".join(
        [
            "_Generated by `scripts/run_resume_benchmarks.sh`._",
            "",
            "**Setup**",
            f"- Run: `{run_name}`",
            f"- Model: `{model_name}`",
            f"- Device: `{device_name}`",
            f"- Dtype: `{dtype}`",
            (
                f"- Software: `Python {python_version}` / `torch {torch_version}` / "
                f"`CUDA {cuda_version}`"
            ),
            (
                f"- Workload: deterministic `{workload}` / `{preset}` trace with "
                f"`output_tokens={output_tokens_label}`"
            ),
            "",
            "**Key Takeaways**",
            *[f"- {takeaway}" for takeaway in takeaways],
            "",
            "**Variant Map**",
            "- `Baseline`: `torch_ref` prefill, `torch_ref` decode, `torch_ref` MLP",
            "- `Triton Prefill + MLP`: `triton` prefill, `torch_ref` decode, `triton` MLP",
            "- `Triton Decode`: `torch_ref` prefill, `triton` decode, `torch_ref` MLP",
            "- `Full Triton`: `triton` prefill, `triton` decode, `triton` MLP",
            "",
            "**Prefill Microbenchmark**",
            "![Prefill latency comparison](docs/images/benchmarks/resume_prefill_latency.svg)",
            "",
            render_prefill_table(prefill_rows),
            "",
            "**Paged Attention Microbenchmark**",
            "![Paged attention latency comparison](docs/images/benchmarks/resume_paged_attention_latency.svg)",
            "",
            render_paged_attention_table(paged_rows),
            "",
            (
                f"**Agent E2E (`{workload}`, `{preset}`, "
                f"`output_tokens={output_tokens_label}`)**"
            ),
            "![Agent e2e TTFT comparison](docs/images/benchmarks/resume_agent_e2e_ttft.svg)",
            "",
            render_e2e_table(e2e_rows),
            "",
            (
                f"**Engine Throughput (`{workload}`, `{preset}`, "
                f"`output_tokens={output_tokens_label}`)**"
            ),
            "![Engine throughput comparison](docs/images/benchmarks/resume_engine_throughput.svg)",
            "",
            render_engine_table(engine_rows),
        ]
    )


def render_local_notes(
    *,
    results_dir: Path,
    takeaways: list[str],
    prefill_rows: list[PrefillSummaryRow],
    e2e_rows: list[E2ESummary],
    engine_rows: list[EngineSummary],
) -> str:
    """Render gitignored resume bullets and metric notes."""
    longest_prefill = max(prefill_rows, key=lambda row: row.prompt_len)
    baseline_e2e = e2e_rows[0]
    best_e2e = min(e2e_rows[1:], key=lambda row: row.mean_ttft_ms)
    ttft_gain = (
        baseline_e2e.mean_ttft_ms - best_e2e.mean_ttft_ms
    ) / baseline_e2e.mean_ttft_ms
    baseline_engine = engine_rows[0]
    best_engine = max(engine_rows[1:], key=lambda row: row.total_tokens_per_s)
    engine_ratio = best_engine.total_tokens_per_s / baseline_engine.total_tokens_per_s

    if engine_ratio >= 0.98:
        engine_bullet = (
            f"- Engine throughput stays close to baseline on the current path "
            f"({baseline_engine.total_tokens_per_s:.2f} tok/s → "
            f"{best_engine.total_tokens_per_s:.2f} tok/s), so it is better used as a caveat "
            "than as a headline metric."
        )
    else:
        engine_bullet = (
            "- Continuous-batching throughput is currently a caveat: Triton-prefill "
            "variants regress sharply on the engine path, so resume bullets should focus "
            "on prefill latency, paged-attention latency, and agent TTFT."
        )

    return "\n".join(
        [
            "# Resume Bullet Drafts",
            "",
            (
                "- Built a minimal LLM inference runtime with paged KV cache, "
                "continuous batching, and a model-adapter-based execution path."
            ),
            (
                f"- Optimized prefill and decode hot paths with Triton kernels; "
                f"cut {longest_prefill.prompt_len}-token prefill latency by "
                f"{(1.0 - 1.0 / longest_prefill.speedup) * 100:.1f}% on local CUDA benchmarks."
            ),
            (
                f"- Validated optimizations with workload-aware evaluation; "
                f"{VARIANT_DISPLAY[best_e2e.variant]} delivered the best agent-session TTFT "
                f"and reduced mean TTFT by {ttft_gain * 100:.1f}% on the current "
                "deterministic trace."
            ),
            engine_bullet,
            "",
            "## Metric Notes",
            "",
            "- `Prefill latency`: isolated prompt ingestion cost before decode begins.",
            "- `Paged attention latency`: raw decode-kernel latency without model overhead.",
            "- `Mean TTFT`: per-turn time-to-first-token on deterministic agent traces.",
            "- `Session Time`: sum of mean per-turn latencies for the one-token agent trace.",
            "- `Engine Total Tok/s`: scheduler + model throughput on the same trace shape.",
            "",
            "## Current Takeaways",
            *[f"- {takeaway}" for takeaway in takeaways],
            "",
            "## Source Run",
            f"- `{results_dir}`",
        ]
    )


def main() -> None:
    """Render charts, README markdown, and local notes."""
    args = parse_args()
    results = load_results(args.results_dir)
    environment = collect_environment(results)
    run_config = collect_run_config(results)
    prefill_rows = summarize_prefill(results)
    paged_rows = summarize_paged_attention(results)
    e2e_rows = summarize_e2e(results)
    engine_rows = summarize_engine(results)
    takeaways = render_key_takeaways(prefill_rows, paged_rows, e2e_rows, engine_rows)

    svg_grouped_bar_chart(
        path=args.charts_dir / "resume_prefill_latency.svg",
        title="Prefill Mean Latency",
        y_label="Mean latency (ms)",
        categories=[f"{row.prompt_len} tok" for row in prefill_rows],
        series=[
            ("Baseline", [row.baseline_ms for row in prefill_rows], "#94a3b8"),
            (
                "Triton Prefill + MLP",
                [row.triton_ms for row in prefill_rows],
                "#2563eb",
            ),
        ],
    )
    svg_grouped_bar_chart(
        path=args.charts_dir / "resume_paged_attention_latency.svg",
        title="Paged Attention Mean Latency",
        y_label="Mean latency (ms)",
        categories=[
            f"b{row.batch_size}/s{row.seq_len}" for row in paged_rows
        ],
        series=[
            ("Baseline", [row.baseline_ms for row in paged_rows], "#94a3b8"),
            ("Triton Decode", [row.triton_ms for row in paged_rows], "#10b981"),
        ],
    )
    svg_grouped_bar_chart(
        path=args.charts_dir / "resume_agent_e2e_ttft.svg",
        title="Agent E2E Mean TTFT",
        y_label="Mean TTFT (ms)",
        categories=[VARIANT_DISPLAY[row.variant] for row in e2e_rows],
        series=[("TTFT", [row.mean_ttft_ms for row in e2e_rows], "#2563eb")],
    )
    svg_grouped_bar_chart(
        path=args.charts_dir / "resume_engine_throughput.svg",
        title="Engine Total Throughput",
        y_label="Total tok/s",
        categories=[VARIANT_DISPLAY[row.variant] for row in engine_rows],
        series=[
            ("Total tok/s", [row.total_tokens_per_s for row in engine_rows], "#7c3aed")
        ],
    )

    readme_block = render_readme_block(
        results_dir=args.results_dir,
        environment=environment,
        run_config=run_config,
        prefill_rows=prefill_rows,
        paged_rows=paged_rows,
        e2e_rows=e2e_rows,
        engine_rows=engine_rows,
    )
    update_readme(args.readme, readme_block)

    args.local_notes.parent.mkdir(parents=True, exist_ok=True)
    args.local_notes.write_text(
        render_local_notes(
            results_dir=args.results_dir,
            takeaways=takeaways,
            prefill_rows=prefill_rows,
            e2e_rows=e2e_rows,
            engine_rows=engine_rows,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
