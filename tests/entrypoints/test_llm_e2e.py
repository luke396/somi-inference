"""End-to-end tests for the high-level LLM API."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from somi_inference.entrypoints.llm import LLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def generate_with_hf(prompt: str, max_new_tokens: int) -> str:
    """Generate reference output with Hugging Face."""
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.size(1) :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@pytest.mark.slow
def test_llm_e2e_greedy_alignment() -> None:
    """Greedy LLM.generate output should match Hugging Face exactly."""
    prompt = "Hello"
    max_new_tokens = 10

    llm = LLM(MODEL_NAME, num_blocks=128)
    somi_output = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    hf_output = generate_with_hf(prompt, max_new_tokens)

    assert somi_output == hf_output, f"somi: {somi_output!r}, hf: {hf_output!r}"


@pytest.mark.slow
def test_llm_e2e_sampling_diversity() -> None:
    """Sampling mode should produce more than one distinct output."""
    llm = LLM(MODEL_NAME, num_blocks=128)

    torch.manual_seed(42)
    outputs = [
        llm.generate("Hello", max_new_tokens=10, temperature=0.8) for _ in range(5)
    ]

    assert len(set(outputs)) >= 2, f"Expected diversity, got: {outputs}"


@pytest.mark.slow
def test_llm_e2e_multiple_requests() -> None:
    """Sequential high-level requests should all complete successfully."""
    llm = LLM(MODEL_NAME, num_blocks=128)

    output_1 = llm.generate("Hello", max_new_tokens=5, temperature=0.0)
    output_2 = llm.generate("World", max_new_tokens=5, temperature=0.0)
    output_3 = llm.generate("Test", max_new_tokens=5, temperature=0.0)

    assert isinstance(output_1, str) and len(output_1) > 0
    assert isinstance(output_2, str) and len(output_2) > 0
    assert isinstance(output_3, str) and len(output_3) > 0
