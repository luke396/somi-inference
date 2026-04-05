"""Tests for benchmark utility helpers."""

import torch

import benchmarks.common as common


def test_collect_environment_metadata_includes_expected_fields(
    monkeypatch,
) -> None:
    """Environment metadata should expose stable JSONL fields."""
    monkeypatch.setattr(common, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(common, "_git_dirty", lambda: True)
    monkeypatch.setattr(common, "_device_name", lambda _device: "Test CPU")
    monkeypatch.setattr(common, "_device_capability", lambda _device: None)

    metadata = common.collect_environment_metadata(torch.device("cpu"))

    assert metadata["git_sha"] == "abc123"
    assert metadata["git_dirty"] is True
    assert metadata["device_type"] == "cpu"
    assert metadata["device_index"] is None
    assert metadata["device_name"] == "Test CPU"
    assert metadata["device_capability"] is None
    assert metadata["torch_version"] == torch.__version__
    assert isinstance(metadata["python_version"], str)
