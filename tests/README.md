# Test Suite Documentation

## Overview

This test suite covers the somi-inference framework from unit tests to end-to-end integration tests.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── core/                    # Core functionality tests
│   ├── test_paged_attention.py
│   └── test_continuous_batching.py
├── models/                  # Model component tests
│   ├── test_qwen2.py       # Base components (RMSNorm, RoPE, attention)
│   ├── test_qwen2_mlp.py
│   ├── test_qwen2_attention.py
│   ├── test_qwen2_decoder_layer.py
│   ├── test_qwen2_model.py
│   ├── test_qwen2_adapter.py
│   ├── test_qwen2_weight_loading.py
│   ├── test_qwen2_e2e.py   # End-to-end tests (slow)
│   ├── test_qwen2_error_handling.py
│   └── test_forward_context.py
└── integration/             # Integration tests
    └── test_paged_attention_integration.py
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=somi_inference --cov-report=html
```

### Run fast tests only (skip slow e2e tests)
```bash
pytest -m "not slow"
```

### Run specific test file
```bash
pytest tests/models/test_qwen2.py -v
```

### Run specific test class
```bash
pytest tests/models/test_qwen2.py::TestRMSNorm -v
```

### Run specific test
```bash
pytest tests/models/test_qwen2.py::TestRMSNorm::test_basic -v
```

### Run integration tests only
```bash
pytest -m integration
```

## Test Markers

- `@pytest.mark.slow`: Tests that take >5 seconds (e.g., HF model loading)
- `@pytest.mark.integration`: Integration tests across multiple components

## Fixtures

### Shared Fixtures (conftest.py)

- `device`: Returns CPU device for testing
- `seed`: Sets random seed to 42 for reproducibility
- `small_model_config`: Small model configuration for unit tests
- `adapter_config`: Configuration for adapter tests
- `make_rope_inputs`: Factory for creating RoPE cos/sin tensors
- `make_forward_context`: Factory for creating ForwardContext
- `make_kv_manager`: Factory for creating KVCacheManager

## Best Practices

1. **Use fixtures**: Prefer fixtures over local helper functions
2. **Parametrize**: Use `@pytest.mark.parametrize` for testing multiple inputs
3. **Descriptive names**: Test names should describe what they verify
4. **Docstrings**: Add docstrings to test classes and complex tests
5. **Markers**: Use markers to categorize tests (slow, integration, etc.)
6. **Assertions**: Use `torch.testing.assert_close` for numerical comparisons
7. **Seeds**: Set random seeds for reproducible tests

## Coverage Goals

- Unit tests: >90% coverage
- Integration tests: Cover critical paths
- E2E tests: Verify correctness against reference implementations
