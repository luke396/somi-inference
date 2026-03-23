"""Tests for Tokenizer."""

import transformers

from somi_inference.tokenizer import Tokenizer


class FakeHFTokenizer:
    """Minimal Hugging Face tokenizer stub."""

    def __init__(self) -> None:
        self.eos_token_id = 151643
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, add_special_tokens):
        """Return deterministic token ids from text."""
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        return [len(text), len(text) + 1]

    def __call__(self, texts, add_special_tokens):
        """Support batch tokenization via tokenizer([...])."""
        return {
            "input_ids": [
                self.encode(text, add_special_tokens=add_special_tokens)
                for text in texts
            ]
        }

    def decode(self, token_ids, skip_special_tokens):
        """Return a deterministic decoded string."""
        self.decode_calls.append(
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
            }
        )
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    def batch_decode(self, token_ids_list, skip_special_tokens):
        """Support batch decoding via the HF tokenizer API."""
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in token_ids_list
        ]


def test_tokenizer_initialization_loads_hf_tokenizer(monkeypatch):
    """Tokenizer should load the backing HF tokenizer from the model name."""
    fake_tokenizer = FakeHFTokenizer()
    loaded_model_names = []

    def fake_from_pretrained(model_name, *args, **kwargs):
        loaded_model_names.append(model_name)
        return fake_tokenizer

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = Tokenizer("Qwen/Qwen2.5-0.5B")

    assert tokenizer.eos_token_id == 151643
    assert loaded_model_names == ["Qwen/Qwen2.5-0.5B"]


def test_tokenizer_encode_disables_special_tokens(monkeypatch):
    """encode should return token ids without implicitly adding special tokens."""
    fake_tokenizer = FakeHFTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _model_name, *args, **kwargs: fake_tokenizer,
    )
    tokenizer = Tokenizer("Qwen/Qwen2.5-0.5B")

    token_ids = tokenizer.encode("Hello")

    assert token_ids == [5, 6]
    assert fake_tokenizer.encode_calls == [
        {
            "text": "Hello",
            "add_special_tokens": False,
        }
    ]


def test_tokenizer_decode_skips_special_tokens(monkeypatch):
    """decode should strip special tokens from the generated text."""
    fake_tokenizer = FakeHFTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _model_name, *args, **kwargs: fake_tokenizer,
    )
    tokenizer = Tokenizer("Qwen/Qwen2.5-0.5B")

    text = tokenizer.decode([11, 12, 13])

    assert text == "decoded:11,12,13"
    assert fake_tokenizer.decode_calls == [
        {
            "token_ids": [11, 12, 13],
            "skip_special_tokens": True,
        }
    ]


def test_tokenizer_batch_encode(monkeypatch):
    """batch_encode should preserve item order and return one list per input."""
    fake_tokenizer = FakeHFTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _model_name, *args, **kwargs: fake_tokenizer,
    )
    tokenizer = Tokenizer("Qwen/Qwen2.5-0.5B")

    token_ids_list = tokenizer.batch_encode(["hi", "world"])

    assert token_ids_list == [[2, 3], [5, 6]]
    assert [call["text"] for call in fake_tokenizer.encode_calls] == ["hi", "world"]


def test_tokenizer_batch_decode(monkeypatch):
    """batch_decode should preserve item order and decode each sequence."""
    fake_tokenizer = FakeHFTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda _model_name, *args, **kwargs: fake_tokenizer,
    )
    tokenizer = Tokenizer("Qwen/Qwen2.5-0.5B")

    texts = tokenizer.batch_decode([[1, 2], [3]])

    assert texts == ["decoded:1,2", "decoded:3"]
