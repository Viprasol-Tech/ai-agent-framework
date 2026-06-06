# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Tests for the LLM abstraction and the deterministic FakeLLM."""

from __future__ import annotations

import pytest

from ai_agent_framework.llm import FakeLLM, LLMClient, Message


def test_message_rejects_invalid_role() -> None:
    with pytest.raises(ValueError):
        Message("robot", "hi")  # type: ignore[arg-type]


def test_fake_llm_is_an_llm_client() -> None:
    assert isinstance(FakeLLM(), LLMClient)


def test_fake_llm_follows_script() -> None:
    llm = FakeLLM(script=["FINAL: scripted reply"])
    assert llm.complete([Message("user", "anything")]) == "FINAL: scripted reply"


def test_fake_llm_routes_arithmetic_to_calculator() -> None:
    reply = FakeLLM().complete([Message("user", "Goal: 2 + 2\nScratchpad:\n")])
    assert reply == "ACTION: calculator(2 + 2)"


def test_fake_llm_finishes_after_observation() -> None:
    prompt = "Goal: 2 + 2\nScratchpad:\nACTION: calculator(2 + 2)\nOBSERVATION: 4\n"
    assert FakeLLM().complete([Message("user", prompt)]) == "FINAL: 4"


def test_fake_llm_echoes_plain_text() -> None:
    reply = FakeLLM().complete([Message("user", "Goal: hello there\nScratchpad:\n")])
    assert reply == "ACTION: echo(hello there)"
