# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Convenience factories for assembling a ready-to-run agent.

These helpers wire the built-in tools and the offline :class:`FakeLLM` together
so callers (and the CLI) can get a working agent in one line, with no API key.
"""

from __future__ import annotations

from ai_agent_framework.agent import Agent
from ai_agent_framework.llm import FakeLLM, LLMClient
from ai_agent_framework.memory import Memory
from ai_agent_framework.tools import (
    CalculatorTool,
    EchoTool,
    MemorySearchTool,
    ToolRegistry,
)


def default_memory() -> Memory:
    """A small seeded memory used by the demo and CLI."""
    memory = Memory()
    memory.add("Viprasol Tech builds fintech and AI agent software.", kind="fact")
    memory.add("The Viprasol AI summit is on March 1st.", kind="fact")
    return memory


def build_agent(llm: LLMClient | None = None, *, memory: Memory | None = None) -> Agent:
    """Construct an agent wired with the three built-in example tools."""
    mem = memory if memory is not None else default_memory()
    registry = ToolRegistry([CalculatorTool(), EchoTool(), MemorySearchTool(mem)])
    return Agent(llm or FakeLLM(), registry, memory=mem)
