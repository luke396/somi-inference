"""Tests for the high-level LLM API."""

from typing import Any, cast

import pytest
import torch

import somi_inference.entrypoints.llm as llm_module
from somi_inference.entrypoints.llm import LLM


@pytest.fixture
def patched_llm_dependencies(monkeypatch):
    """Patch all LLM collaborators with lightweight test doubles."""
    registry: dict[str, Any] = {
        "run_calls": [],
        "outputs_by_seq_id": {
            0: [101, 102],
            1: [201],
        },
    }

    class FakeTorchModule(torch.nn.Module):
        """Torch module stub used to verify device and mode setup."""

        def __init__(self):
            super().__init__()
            self.to_kwargs = None
            self.eval_called = False
            self.requires_grad_value = None

        def to(self, *args, **kwargs):
            self.to_kwargs = kwargs
            return self

        def eval(self):
            self.eval_called = True
            return self

        def requires_grad_(self, requires_grad=True):
            self.requires_grad_value = requires_grad
            return self

    class FakeAdapter:
        """Adapter stub that exposes a torch.nn.Module model."""

        def __init__(self):
            self.model = FakeTorchModule()
            self.prefill_attention_backend = "torch_ref"
            self.decode_attention_backend = "torch_ref"
            self.mlp_backend = "torch_ref"

    adapter = FakeAdapter()
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

    class FakeKVCacheManager:
        """KV cache stub that accepts flexible constructor kwargs."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            registry["kv_manager"] = self

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
                self.scheduler = args[1]
                self.eos_token_id = (
                    args[2] if len(args) > 2 else kwargs["eos_token_id"]
                )
            else:
                self.model_runner = kwargs["model_runner"]
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
    registry["model"] = adapter.model
    return registry


def test_llm_initialization_wires_phase2_components(patched_llm_dependencies):
    """LLM should wire tokenizer, KV cache, scheduler, engine, and runner."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    assert llm.tokenizer is patched_llm_dependencies["tokenizer"]
    assert llm.kv_manager is patched_llm_dependencies["kv_manager"]
    assert llm.engine is patched_llm_dependencies["engine"]
    assert llm._next_seq_id == 0

    kv_kwargs = patched_llm_dependencies["kv_manager"].kwargs
    assert kv_kwargs["num_blocks"] == 128
    assert kv_kwargs["num_kv_heads"] == 2
    assert kv_kwargs["head_dim"] == 64
    assert kv_kwargs["n_layers"] == 24

    runner = patched_llm_dependencies["model_runner"]
    assert runner.adapter is patched_llm_dependencies["adapter"]
    assert runner.sampler is patched_llm_dependencies["sampler"]
    assert runner.kv_manager is patched_llm_dependencies["kv_manager"]


def test_llm_initialization_defaults_to_cuda_when_available(
    monkeypatch,
    patched_llm_dependencies,
):
    """LLM should prefer CUDA by default when torch reports it is available."""
    monkeypatch.setattr(llm_module.torch.cuda, "is_available", lambda: True)

    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    assert llm.device == torch.device("cuda")
    kv_kwargs = patched_llm_dependencies["kv_manager"].kwargs
    assert kv_kwargs["device"] == torch.device("cuda")
    assert patched_llm_dependencies["model"].to_kwargs["device"] == torch.device("cuda")


def test_llm_initialization_forwards_device_and_dtype(patched_llm_dependencies):
    """LLM should forward device and dtype to the KV cache manager."""
    llm = LLM(
        "Qwen/Qwen2.5-0.5B",
        num_blocks=128,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert llm.device == torch.device("cpu")
    assert llm.dtype == torch.bfloat16
    kv_kwargs = patched_llm_dependencies["kv_manager"].kwargs
    assert kv_kwargs["device"] == torch.device("cpu")
    assert kv_kwargs["dtype"] == torch.bfloat16
    model = patched_llm_dependencies["model"]
    assert model.to_kwargs == {
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
    }
    assert model.eval_called is True
    assert model.requires_grad_value is False


def test_llm_initialization_forwards_prefill_attention_backend(
    patched_llm_dependencies,
):
    """LLM should forward the requested prefill attention backend to adapters."""
    LLM(
        "Qwen/Qwen2.5-0.5B",
        num_blocks=128,
        prefill_attention_backend="torch_ref",
    )

    adapter = patched_llm_dependencies["adapter"]
    assert adapter.prefill_attention_backend == "torch_ref"


def test_llm_initialization_forwards_mlp_backend(
    patched_llm_dependencies,
):
    """LLM should forward the requested MLP backend to adapters."""
    LLM(
        "Qwen/Qwen2.5-0.5B",
        num_blocks=128,
        mlp_backend="torch_ref",
    )

    adapter = patched_llm_dependencies["adapter"]
    assert adapter.mlp_backend == "torch_ref"


def test_llm_initialization_forwards_decode_attention_backend(
    patched_llm_dependencies,
):
    """LLM should forward the requested decode attention backend to adapters."""
    LLM(
        "Qwen/Qwen2.5-0.5B",
        num_blocks=128,
        decode_attention_backend="torch_ref",
    )

    adapter = patched_llm_dependencies["adapter"]
    assert adapter.decode_attention_backend == "torch_ref"


def test_llm_generate_builds_request_and_decodes_generated_tokens(
    patched_llm_dependencies,
):
    """Generate should enqueue one request and decode only generated tokens."""
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


def test_llm_generate_reuses_real_engine_without_finished_leak(monkeypatch):
    """Repeated generate calls should return only the current request output."""
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_size": 896,
    }

    class FakeTokenizer:
        def __init__(self, model_name, *args, **kwargs):
            self.model_name = model_name
            self.eos_token_id = 151643

        def encode(self, text):
            return [len(text), len(text) + 1]

        def decode(self, token_ids):
            return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    class FakeSampler:
        def __init__(self):
            pass

    class FakeKVCacheManager:
        def __init__(self, *args, **kwargs):
            self.registered_seq_ids = []
            self.freed_seq_ids = []
            self.kv_caches = [
                type("FakeCache", (), {"key_cache": torch.zeros(1)})(),
            ]

        def register_sequence(self, seq_id):
            self.registered_seq_ids.append(seq_id)

        def free_sequence(self, seq_id):
            self.freed_seq_ids.append(seq_id)

    class FakeModelRunner:
        def __init__(self, *args, **kwargs):
            if args:
                self.kv_manager = args[2]
            else:
                self.kv_manager = kwargs["kv_manager"]

        def prefill(self, input_ids, seq_id, sampling_params):
            del input_ids, sampling_params
            return 100 + seq_id

        def decode(self, input_ids, seq_ids, sampling_params, token_histories):
            del input_ids, seq_ids, sampling_params, token_histories
            message = "decode should not be called for max_new_tokens=1"
            raise AssertionError(message)

    monkeypatch.setattr(
        llm_module,
        "load_model",
        lambda _model_name, *args, **kwargs: (object(), config),
    )
    monkeypatch.setattr(llm_module, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(llm_module, "KVCacheManager", FakeKVCacheManager)
    monkeypatch.setattr(llm_module, "Sampler", FakeSampler)
    monkeypatch.setattr(llm_module, "ModelRunner", FakeModelRunner)

    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    output_0 = llm.generate("Hello", max_new_tokens=1, temperature=0.0)
    output_1 = llm.generate("World", max_new_tokens=1, temperature=0.0)

    assert output_0 == "decoded:100"
    assert output_1 == "decoded:101"
    kv_manager = cast("FakeKVCacheManager", llm.kv_manager)
    assert kv_manager.registered_seq_ids == [0, 1]
    assert kv_manager.freed_seq_ids == [0, 1]


def test_llm_generate_stream_not_implemented(patched_llm_dependencies):
    """Streaming remains explicitly out of scope for Phase 2."""
    llm = LLM("Qwen/Qwen2.5-0.5B", num_blocks=128)

    with pytest.raises(NotImplementedError, match="Streaming not yet implemented"):
        list(llm.generate_stream("Hello", max_new_tokens=5))
