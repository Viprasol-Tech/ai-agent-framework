# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""End-to-end tests for the agent control loop, all running offline."""

from __future__ import annotations

import pytest

from ai_agent_framework.agent import Agent
from ai_agent_framework.llm import FakeLLM, LLMClient, Message
from ai_agent_framework.memory import Memory
from ai_agent_framework.tools import (
    CalculatorTool,
    EchoTool,
    MemorySearchTool,
    ToolRegistry,
)


def _agent(script: list[str] | None = None, memory: Memory | None = None) -> Agent:
    mem = memory if memory is not None else Memory()
    registry = ToolRegistry([CalculatorTool(), EchoTool(), MemorySearchTool(mem)])
    return Agent(FakeLLM(script=script), registry, memory=mem)


def test_agent_solves_arithmetic_with_calculator() -> None:
    result = _agent().run("2 * (3 + 4)")
    assert result.completed is True
    assert result.answer == "14"
    assert any(step.tool == "calculator" for step in result.steps)


def test_agent_echoes_plain_request() -> None:
    result = _agent().run("hello world")
    assert result.answer == "hello world"


def test_agent_recalls_from_memory() -> None:
    memory = Memory()
    memory.add("The launch date is March 1st.")
    result = _agent(memory=memory).run("recall the launch date")
    assert "March" in result.answer


def test_scripted_action_then_final() -> None:
    agent = _agent(script=["ACTION: calculator(6 * 7)", "FINAL: the answer is 42"])
    result = agent.run("compute the meaning of life")
    assert result.answer == "the answer is 42"
    assert result.steps[0].tool == "calculator"
    assert result.steps[0].observation == "42"


def test_unknown_tool_yields_error_observation() -> None:
    agent = _agent(script=["ACTION: nope(x)", "FINAL: done"])
    result = agent.run("use a missing tool")
    assert "unknown tool" in (result.steps[0].observation or "")
    assert result.answer == "done"


def test_malformed_reply_is_reported() -> None:
    agent = _agent(script=["not a valid line", "FINAL: ok"])
    result = agent.run("anything")
    assert "must be" in (result.steps[0].observation or "")
    assert result.answer == "ok"


class _NeverFinishingLLM(LLMClient):
    """An LLM that always asks for the same tool, never finishing."""

    def complete(self, messages: list[Message]) -> str:
        return "ACTION: echo(loop)"


def test_loop_stops_at_max_steps() -> None:
    agent = Agent(_NeverFinishingLLM(), ToolRegistry([EchoTool()]), max_steps=3)
    result = agent.run("loop forever")
    assert result.completed is False
    assert result.status == "max_steps"
    assert len(result.steps) == 3


def test_invalid_max_steps_raises() -> None:
    with pytest.raises(ValueError):
        Agent(FakeLLM(), max_steps=0)


def test_agent_accepts_list_of_tools() -> None:
    agent = Agent(FakeLLM(), [EchoTool()])
    assert agent.tools.has("echo")


def test_memory_records_goal_and_answer() -> None:
    memory = Memory()
    _agent(memory=memory).run("2 + 2")
    kinds = {record.kind for record in memory}
    assert "goal" in kinds
    assert "answer" in kinds
