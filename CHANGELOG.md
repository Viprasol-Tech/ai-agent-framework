# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-06-06

### Added
- Initial public release of **AI Agent Framework** by Viprasol Tech.
- `Agent` — a bounded `perceive → think → act` (ReAct-style) control loop.
- `Tool` protocol with `ToolResult`, `ToolError`, and a `ToolRegistry`.
- Three real example tools: a safe AST `calculator`, `echo`, and `memory_search`.
- `Memory` store with deterministic keyword search.
- `LLMClient` abstraction, a deterministic offline `FakeLLM`, and an optional,
  lazily-imported `OpenAIClient` (no hard dependency).
- Zero-dependency `argparse` CLI (`agent-framework`) and a runnable demo.
- Tooling: `ruff`, `mypy --strict`, `pytest` (35 tests), GitHub Actions CI
  across Linux/macOS/Windows × Python 3.11–3.13, and a slim Docker image.

[0.1.0]: https://github.com/Viprasol-Tech/ai-agent-framework/releases/tag/v0.1.0
