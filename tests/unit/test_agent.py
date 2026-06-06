# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Smoke tests for the public package surface.

The full behavioural suite lives in the top-level ``tests/`` modules; this file
keeps a couple of import/version smoke checks for the installed package.
"""

from __future__ import annotations

import ai_agent_framework as af
from ai_agent_framework import Agent, FakeLLM


def test_version_is_exposed() -> None:
    assert af.__version__ == "0.1.0"


def test_public_api_is_importable() -> None:
    for name in ("Agent", "FakeLLM", "Memory", "Tool", "ToolRegistry", "CalculatorTool"):
        assert hasattr(af, name)


def test_agent_constructs_without_tools() -> None:
    agent = Agent(FakeLLM())
    assert agent.max_steps == 6
