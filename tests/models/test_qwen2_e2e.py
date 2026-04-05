"""End-to-end test: somi greedy decode matches HF model.generate()."""

import pytest
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.mark.slow
class TestQwen2E2E:
    def test_greedy_decode_matches_hf(self):
        """Somi adapter greedy decode output matches HF generate() exactly."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from somi_inference.core.paged_attention import KVCacheManager
        from somi_inference.models.qwen2_adapter import load_from_hf

        # Load HF model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.float32,
        )
        hf_model.requires_grad_(False)

        # Load somi model
        adapter = load_from_hf(MODEL_NAME)

        # Tokenize prompt
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # (1, prompt_len)
        prompt_len = input_ids.shape[1]
        max_new_tokens = 20

        # HF greedy decode
        with torch.inference_mode():
            hf_output = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        hf_tokens = hf_output[0, prompt_len:].tolist()

        # Somi greedy decode
        hf_config = hf_model.config
        kv_manager = KVCacheManager(
            num_blocks=256,
            block_size=16,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
            n_layers=hf_config.num_hidden_layers,
        )
        kv_manager.register_sequence(0)

        somi_tokens = []
        with torch.inference_mode():
            # Prefill
            logits = adapter.prefill(input_ids, kv_manager, seq_id=0)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            somi_tokens.append(next_token)

            # Decode
            for _ in range(max_new_tokens - 1):
                token_input = torch.tensor([[next_token]])
                logits = adapter.decode(token_input, kv_manager, seq_ids=[0])
                next_token = logits[:, 0, :].argmax(dim=-1).item()
                somi_tokens.append(next_token)

        # Compare
        assert somi_tokens == hf_tokens, (
            f"Token mismatch!\n"
            f"HF:   {hf_tokens}\n"
            f"Somi: {somi_tokens}\n"
            f"HF text:   {tokenizer.decode(hf_tokens)}\n"
            f"Somi text: {tokenizer.decode(somi_tokens)}"
        )
