# TDD Test Suites

This directory holds forward-looking TDD tests that are not yet expected to
pass in the default committed test suite.

Current layout:

```text
tests_tdd/
└── phase2/   # ModelRunner / Sampler / Tokenizer / Loader / LLM API tests
```

Run these suites explicitly while developing the corresponding phase:

```bash
uv run pytest tests_tdd/phase2
```
