"""Tests for the high-level LLM API."""

import pytest

import somi_inference.entrypoints.llm as llm_module
from somi_inference.entrypoints.llm import LLM


@pytest.fixture
def patched_llm_dependencies(monkeypatch):
    """Patch all LLM collaborators with lightweight test doubles."""
    registry = {
        "run_calls": [],
        "outputs_by_seq_id": {
            0: [101, 102],
            1: [201],
        },
    }
    adapter = object()
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_size": 896,
    }

    class FakeTokenizer:
        """Tokenizer stub used by LLM tests."""

        def __init__(self, model_name, *args, **kwargs):
            self.model_name = model_name
            self.eos_token_id = 151643
            self.encode_calls = []
            self.decode_calls = []
            registry["tokenizer"] = self

        def encode(self, text):
            self.encode_calls.append(text)
            return [len(text), len(text) + 1]

        def decode(self, token_ids):
            token_ids = list(token_ids)
            self.decode_calls.append(token_ids)
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    class FakeAllocator:
        def num_free_blocks(self):
            return 128

    class FakeKVCacheManager:
        """KV cache stub that accepts flexible constructor kwargs."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.allocator = FakeAllocator()
            registry["kv_manager"] = self

        def get_num_free_blocks(self):
            return self.allocator.num_free_blocks()

    class FakeSampler:
        def __init__(self):
            registry["sampler"] = self

    class FakeModelRunner:
        """ModelRunner stub used to verify LLM wiring."""

        def __init__(self, *args, **kwargs):
            if args:
                self.adapter = args[0]
                self.sampler = args[1]
                self.kv_manager = args[2]
            else:
                self.adapter = kwargs["adapter"]
                self.sampler = kwargs["sampler"]
                self.kv_manager = kwargs["kv_manager"]
            registry["model_runner"] = self

    class FakeScheduler:
        """Scheduler stub that records initialization."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            registry["scheduler"] = self

    class FakeEngine:
        """Engine stub that returns pre-baked generated token ids."""

        def __init__(self, *args, **kwargs):
            if args:
                self.model_runner = args[0]
                self.kv_manager = args[1]
                self.scheduler = args[2]
                self.eos_token_id = args[3] if len(args) > 3 else kwargs["eos_token_id"]
            else:
                self.model_runner = kwargs.get("model_runner") or kwargs.get("model")
                self.kv_manager = kwargs["kv_manager"]
                self.scheduler = kwargs["scheduler"]
                self.eos_token_id = kwargs["eos_token_id"]
            registry["engine"] = self

        def run(self, requests):
            snapshot = []
            finished = []
            for arrival_step, seq in list(requests):
                snapshot.append(
                    {
                        "arrival_step": arrival_step,
                        "seq_id": seq.seq_id,
                        "prompt_tokens": list(seq.prompt_tokens),
                        "max_new_tokens": seq.max_new_tokens,
                        "sampling_params": seq.sampling_params,
                    }
                )
                seq.output_tokens = list(registry["outputs_by_seq_id"][seq.seq_id])
                finished.append(seq)
            registry["run_calls"].append(snapshot)
            return finished

    monkeypatch.setattr(
        llm_module,
        "load_model",
        lambda _model_name, *args, **kwargs: (adapter, config),
    )
    monkeypatch.setattr(llm_module, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(llm_module, "KVCacheManager", FakeKVCacheManager)
    monkeypatch.setattr(llm_module, "Sampler", FakeSampler)
    monkeypatch.setattr(llm_module, "ModelRunner", FakeModelRunner)
    monkeypatch.setattr(llm_module, "Scheduler", FakeScheduler)
    monkeypatch.setattr(llm_module, "ContinuousBatchingEngine", FakeEngine)
    registry["adapter"] = adapter
    registry["config"] = config
    return registry


def test_llm_initialization_wires_phase2_components(patched_llm_dependencies):
    """LLM should wire tokenizer, KV cache, scheduler, engine, and runner."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    assert llm.tokenizer is patched_llm_dependencies["tokenizer"]
    assert llm.kv_manager is patched_llm_dependencies["kv_manager"]
    assert llm.engine is patched_llm_dependencies["engine"]
    assert llm._next_seq_id == 0

    kv_kwargs = patched_llm_dependencies["kv_manager"].kwargs
    layer_count = kv_kwargs.get("n_layers", kv_kwargs.get("num_layers"))
    assert kv_kwargs["num_blocks"] == 128
    assert kv_kwargs["num_kv_heads"] == 2
    assert kv_kwargs["head_dim"] == 64
    assert layer_count == 24

    runner = patched_llm_dependencies["model_runner"]
    assert runner.adapter is patched_llm_dependencies["adapter"]
    assert runner.sampler is patched_llm_dependencies["sampler"]
    assert runner.kv_manager is patched_llm_dependencies["kv_manager"]


def test_llm_generate_builds_request_and_decodes_generated_tokens(
    patched_llm_dependencies,
):
    """generate should enqueue one request and decode only the generated tokens."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output = llm.generate(
        "Hello",
        max_new_tokens=5,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    assert output == "decoded:101,102"
    request = patched_llm_dependencies["run_calls"][0][0]
    assert request["arrival_step"] == 0
    assert request["seq_id"] == 0
    assert request["prompt_tokens"] == [5, 6]
    assert request["max_new_tokens"] == 5

    params = request["sampling_params"]
    assert params.temperature == 0.8
    assert params.top_k == 50
    assert params.top_p == 0.9
    assert params.repetition_penalty == 1.2

    assert patched_llm_dependencies["tokenizer"].decode_calls == [[101, 102]]
    assert llm._next_seq_id == 1


def test_llm_generate_multiple_calls_increment_sequence_ids(
    patched_llm_dependencies,
):
    """Each generate call should use a fresh sequence ID."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output_0 = llm.generate("Hello", max_new_tokens=2, temperature=0.0)
    output_1 = llm.generate("World", max_new_tokens=1, temperature=0.0)

    assert output_0 == "decoded:101,102"
    assert output_1 == "decoded:201"
    assert patched_llm_dependencies["run_calls"][0][0]["seq_id"] == 0
    assert patched_llm_dependencies["run_calls"][1][0]["seq_id"] == 1
    assert llm._next_seq_id == 2


def test_llm_generate_stream_not_implemented(patched_llm_dependencies):
    """Streaming remains explicitly out of scope for Phase 2."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    with pytest.raises(NotImplementedError, match="Streaming not yet implemented"):
        list(llm.generate_stream("Hello", max_new_tokens=5))
