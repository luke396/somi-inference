"""Deterministic workload builders for benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

WorkloadName = Literal["agent-session", "chat-serving"]
PresetName = Literal["short", "mid", "long"]
DEFAULT_BASE_PROMPT_SEED = "Explain how paged attention works in one paragraph."


class BenchmarkTokenizer(Protocol):
    """Tokenizer surface used by synthetic workload builders."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back into text."""
        ...


@dataclass(frozen=True)
class WorkloadPromptSeeds:
    """Prompt seeds used to synthesize one workload family."""

    user: str
    assistant: str
    tool: str | None = None


@dataclass(frozen=True)
class WorkloadPresetSpec:
    """Deterministic session structure for one workload preset."""

    base_prompt_tokens: int
    user_tokens_per_turn: tuple[int, ...]
    tool_tokens_per_turn: tuple[int, ...]
    output_token_options: tuple[int, ...]


@dataclass(frozen=True)
class WorkloadTurnInput:
    """Prebuilt prompt sections for one turn before assistant output is appended."""

    user_tokens: int
    tool_tokens: int
    user_section: str
    tool_section: str | None


@dataclass(frozen=True)
class WorkloadTurnCase:
    """One deterministic workload turn to benchmark."""

    workload: WorkloadName
    preset: PresetName
    scenario: str
    session_id: str
    turn_idx: int
    num_turns: int
    base_prompt_tokens: int
    user_tokens: int
    tool_tokens: int
    requested_prompt_tokens: int
    actual_prompt_tokens: int
    requested_output_tokens: int
    prompt_text: str


WORKLOAD_PROMPT_SEEDS: dict[WorkloadName, WorkloadPromptSeeds] = {
    "agent-session": WorkloadPromptSeeds(
        user="Review the latest state, choose the next action, and explain it briefly.",
        assistant="Next action, rationale, and short follow-up plan.",
        tool=(
            "Tool output with logs, code snippets, stack traces, and structured "
            "results."
        ),
    ),
    "chat-serving": WorkloadPromptSeeds(
        user=(
            "Continue the conversation clearly, answer the question, and preserve "
            "context."
        ),
        assistant="A direct answer with context, detail, and examples.",
    ),
}

WORKLOAD_PRESETS: dict[WorkloadName, dict[PresetName, WorkloadPresetSpec]] = {
    "agent-session": {
        "short": WorkloadPresetSpec(
            base_prompt_tokens=128,
            user_tokens_per_turn=(48, 48, 48, 48, 48, 48),
            tool_tokens_per_turn=(0, 0, 0, 0, 0, 256),
            output_token_options=(1, 32),
        ),
        "mid": WorkloadPresetSpec(
            base_prompt_tokens=128,
            user_tokens_per_turn=(64, 64, 64, 64, 64, 64),
            tool_tokens_per_turn=(0, 256, 0, 256, 0, 0),
            output_token_options=(1, 32),
        ),
        "long": WorkloadPresetSpec(
            base_prompt_tokens=128,
            user_tokens_per_turn=(96, 96, 96, 96, 96, 96),
            tool_tokens_per_turn=(256, 512, 0, 512, 0, 0),
            output_token_options=(1, 32),
        ),
    },
    "chat-serving": {
        "short": WorkloadPresetSpec(
            base_prompt_tokens=128,
            user_tokens_per_turn=(32, 32, 32, 32, 32, 32),
            tool_tokens_per_turn=(0, 0, 0, 0, 0, 0),
            output_token_options=(64, 128, 256),
        ),
        "mid": WorkloadPresetSpec(
            base_prompt_tokens=256,
            user_tokens_per_turn=(48, 48, 48, 48, 48, 48),
            tool_tokens_per_turn=(0, 0, 0, 0, 0, 0),
            output_token_options=(64, 128, 256),
        ),
        "long": WorkloadPresetSpec(
            base_prompt_tokens=512,
            user_tokens_per_turn=(64, 64, 64, 64, 64, 64),
            tool_tokens_per_turn=(0, 0, 0, 0, 0, 0),
            output_token_options=(64, 128, 256),
        ),
    },
}


def make_target_prompt(
    tokenizer: BenchmarkTokenizer,
    base_prompt: str,
    target_tokens: int,
) -> str:
    """Build a prompt string whose encoded length closely tracks `target_tokens`."""
    if target_tokens <= 0:
        message = "Target token counts must be positive."
        raise ValueError(message)
    base_token_ids = tokenizer.encode(base_prompt)
    if not base_token_ids:
        message = "Prompt text produced zero tokens; provide a non-empty prompt."
        raise ValueError(message)
    target_decode_len = target_tokens
    best_prompt = base_prompt
    best_distance = float("inf")

    for _ in range(8):
        repeats = (target_decode_len + len(base_token_ids) - 1) // len(base_token_ids)
        repeated_ids = (base_token_ids * repeats)[:target_decode_len]
        prompt_text = tokenizer.decode(repeated_ids)
        actual_tokens = len(tokenizer.encode(prompt_text))
        distance = abs(actual_tokens - target_tokens)
        if distance < best_distance:
            best_prompt = prompt_text
            best_distance = distance
        if actual_tokens == target_tokens:
            return prompt_text
        target_decode_len = max(target_decode_len + (target_tokens - actual_tokens), 1)

    return best_prompt


def _format_role_segment(role: str, body: str) -> str:
    """Wrap a synthetic segment with a stable role header."""
    return f"<{role}>\n{body}"


def _build_workload_turn_inputs(
    *,
    tokenizer: BenchmarkTokenizer,
    prompt_seeds: WorkloadPromptSeeds,
    preset_spec: WorkloadPresetSpec,
) -> list[WorkloadTurnInput]:
    """Prebuild the user/tool prompt sections shared across output variants."""
    if len(preset_spec.user_tokens_per_turn) != len(preset_spec.tool_tokens_per_turn):
        message = "Workload preset user/tool turn counts must match."
        raise ValueError(message)

    turn_inputs: list[WorkloadTurnInput] = []
    for user_tokens, tool_tokens in zip(
        preset_spec.user_tokens_per_turn,
        preset_spec.tool_tokens_per_turn,
        strict=True,
    ):
        user_section = _format_role_segment(
            "user",
            make_target_prompt(tokenizer, prompt_seeds.user, user_tokens),
        )
        tool_section = None
        if tool_tokens > 0:
            tool_prompt = prompt_seeds.tool
            if tool_prompt is None:
                message = "Workload preset requested tool tokens without a tool seed."
                raise ValueError(message)
            tool_section = _format_role_segment(
                "tool",
                make_target_prompt(tokenizer, tool_prompt, tool_tokens),
            )
        turn_inputs.append(
            WorkloadTurnInput(
                user_tokens=user_tokens,
                tool_tokens=tool_tokens,
                user_section=user_section,
                tool_section=tool_section,
            )
        )
    return turn_inputs


def build_workload_turn_cases(
    *,
    tokenizer: BenchmarkTokenizer,
    workload: WorkloadName,
    preset: PresetName,
    base_prompt_seed: str,
) -> list[WorkloadTurnCase]:
    """Build deterministic multi-turn workload cases for one preset."""
    prompt_seeds = WORKLOAD_PROMPT_SEEDS[workload]
    preset_spec = WORKLOAD_PRESETS[workload][preset]
    turn_inputs = _build_workload_turn_inputs(
        tokenizer=tokenizer,
        prompt_seeds=prompt_seeds,
        preset_spec=preset_spec,
    )
    base_prompt = make_target_prompt(
        tokenizer,
        base_prompt_seed,
        preset_spec.base_prompt_tokens,
    )
    base_section = _format_role_segment("system", base_prompt)
    num_turns = len(turn_inputs)
    cases: list[WorkloadTurnCase] = []

    for output_tokens in preset_spec.output_token_options:
        scenario = f"{workload}:{preset}:out{output_tokens}"
        session_id = f"{workload}-{preset}-out{output_tokens}"
        assistant_section = _format_role_segment(
            "assistant",
            make_target_prompt(tokenizer, prompt_seeds.assistant, output_tokens),
        )
        history_sections = [base_section]
        requested_history_tokens = preset_spec.base_prompt_tokens

        for turn_idx, turn_input in enumerate(turn_inputs, start=1):
            current_sections = [*history_sections, turn_input.user_section]
            if turn_input.tool_section is not None:
                current_sections.append(turn_input.tool_section)

            prompt_text = "\n\n".join(current_sections)
            requested_prompt_tokens = (
                requested_history_tokens
                + turn_input.user_tokens
                + turn_input.tool_tokens
            )
            actual_prompt_tokens = len(tokenizer.encode(prompt_text))
            cases.append(
                WorkloadTurnCase(
                    workload=workload,
                    preset=preset,
                    scenario=scenario,
                    session_id=session_id,
                    turn_idx=turn_idx,
                    num_turns=num_turns,
                    base_prompt_tokens=preset_spec.base_prompt_tokens,
                    user_tokens=turn_input.user_tokens,
                    tool_tokens=turn_input.tool_tokens,
                    requested_prompt_tokens=requested_prompt_tokens,
                    actual_prompt_tokens=actual_prompt_tokens,
                    requested_output_tokens=output_tokens,
                    prompt_text=prompt_text,
                )
            )
            history_sections = [*current_sections, assistant_section]
            requested_history_tokens = requested_prompt_tokens + output_tokens

    return cases


def filter_turn_cases_by_output_tokens(
    turn_cases: list[WorkloadTurnCase],
    output_tokens: tuple[int, ...] | None,
) -> list[WorkloadTurnCase]:
    """Filter workload cases by requested output tokens when requested."""
    if output_tokens is None:
        return turn_cases

    allowed_output_tokens = set(output_tokens)
    filtered_cases = [
        case
        for case in turn_cases
        if case.requested_output_tokens in allowed_output_tokens
    ]
    if not filtered_cases:
        message = (
            "No workload cases matched --output-tokens="
            f"{sorted(allowed_output_tokens)}."
        )
        raise ValueError(message)
    return filtered_cases
