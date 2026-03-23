"""End-to-end tests for LLM with HuggingFace alignment."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from somi_inference.entrypoints.llm import LLM


def generate_with_hf(prompt: str, max_new_tokens: int) -> str:
    """Generate with HuggingFace for reference."""
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only new tokens
    new_tokens = output_ids[0, input_ids.size(1) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# Task 17: E2E Tests with HF Alignment
# ============================================================================


@pytest.mark.slow
def test_llm_e2e_greedy_alignment():
    """LLM greedy decoding should align with HF"""
    prompt = "Hello"
    max_new_tokens = 10

    # Generate with somi
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)
    somi_output = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)

    # Generate with HF
    hf_output = generate_with_hf(prompt, max_new_tokens)

    # Should match exactly for greedy decoding
    assert somi_output == hf_output, f"somi: {somi_output!r}, hf: {hf_output!r}"


@pytest.mark.slow
def test_llm_e2e_sampling_diversity():
    """LLM sampling should produce diverse outputs"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    torch.manual_seed(42)
    outputs = [
        llm.generate("Hello", max_new_tokens=10, temperature=0.8) for _ in range(5)
    ]

    # Should have at least 2 different outputs
    unique_outputs = set(outputs)
    assert len(unique_outputs) >= 2, f"Expected diversity, got: {outputs}"


@pytest.mark.slow
def test_llm_e2e_multiple_requests():
    """LLM should handle multiple sequential requests"""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    # Generate multiple times
    output1 = llm.generate("Hello", max_new_tokens=5, temperature=0.0)
    output2 = llm.generate("World", max_new_tokens=5, temperature=0.0)
    output3 = llm.generate("Test", max_new_tokens=5, temperature=0.0)

    assert isinstance(output1, str) and len(output1) > 0
    assert isinstance(output2, str) and len(output2) > 0
    assert isinstance(output3, str) and len(output3) > 0
