# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Tests for the argparse-based command-line interface."""

from __future__ import annotations

import pytest

from ai_agent_framework.cli import main
from ai_agent_framework.providers import build_agent


def test_cli_runs_calculator(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "2 * (3 + 4)"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip().endswith("14")


def test_cli_verbose_prints_steps(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "10 - 1", "--verbose"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "calculator" in out
    assert out.strip().endswith("9")


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "ai-agent-framework" in capsys.readouterr().out


def test_build_agent_has_three_tools() -> None:
    agent = build_agent()
    assert set(agent.tools.names()) == {"calculator", "echo", "memory_search"}
