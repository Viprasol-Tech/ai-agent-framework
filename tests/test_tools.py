# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Tests for the built-in tools, the Tool protocol, and the registry."""

from __future__ import annotations

import pytest

from ai_agent_framework.memory import Memory
from ai_agent_framework.tools import (
    CalculatorTool,
    EchoTool,
    MemorySearchTool,
    Tool,
    ToolRegistry,
    ToolResult,
)


def test_echo_tool_round_trips() -> None:
    tool = EchoTool()
    assert isinstance(tool, Tool)
    assert tool.run("ping") == ToolResult(output="ping", ok=True)


def test_calculator_evaluates_expression() -> None:
    tool = CalculatorTool()
    assert tool.run("2 * (3 + 4)").output == "14"
    assert tool.run("10 / 4").output == "2.5"
    assert tool.run("-2 ** 3").output == "-8"


def test_calculator_rejects_bad_input() -> None:
    result = CalculatorTool().run("__import__('os')")
    assert result.ok is False
    assert "error" in result.output


def test_calculator_handles_division_by_zero() -> None:
    assert CalculatorTool().run("1 / 0").ok is False


def test_memory_search_tool_finds_records() -> None:
    memory = Memory()
    memory.add("Viprasol builds AI agents")
    tool = MemorySearchTool(memory)
    assert "Viprasol" in tool.run("agents").output


def test_memory_search_tool_handles_no_match() -> None:
    result = MemorySearchTool(Memory()).run("nothing")
    assert result.ok is True
    assert "No memory found" in result.output


def test_registry_register_get_and_catalogue() -> None:
    registry = ToolRegistry([CalculatorTool(), EchoTool()])
    assert registry.has("calculator")
    assert registry.names() == ["calculator", "echo"]
    assert "calculator" in registry.catalogue()


def test_registry_duplicate_raises() -> None:
    registry = ToolRegistry([EchoTool()])
    with pytest.raises(ValueError):
        registry.register(EchoTool())
