# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""AI Agent Framework — a minimal, dependency-light toolkit for building LLM agents.

The framework provides four small, composable building blocks:

* :class:`~ai_agent_framework.agent.Agent` — a ``perceive -> think -> act`` loop.
* :class:`~ai_agent_framework.tools.Tool` — a pluggable tool protocol with examples.
* :class:`~ai_agent_framework.memory.Memory` — a lightweight memory store.
* :class:`~ai_agent_framework.llm.LLMClient` — an LLM abstraction with a built-in
  deterministic :class:`~ai_agent_framework.llm.FakeLLM` so everything runs offline.
"""

from __future__ import annotations

from ai_agent_framework.agent import Agent, AgentResult, Step
from ai_agent_framework.llm import FakeLLM, LLMClient, Message
from ai_agent_framework.memory import Memory, MemoryRecord
from ai_agent_framework.tools import (
    CalculatorTool,
    EchoTool,
    MemorySearchTool,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
)

__version__ = "0.1.0"
__author__ = "Viprasol Tech Private Limited"

__all__ = [
    "Agent",
    "AgentResult",
    "CalculatorTool",
    "EchoTool",
    "FakeLLM",
    "LLMClient",
    "Memory",
    "MemoryRecord",
    "MemorySearchTool",
    "Message",
    "Step",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "__version__",
]
