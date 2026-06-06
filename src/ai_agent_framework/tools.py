# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""The pluggable tool protocol, a registry, and three real example tools.

A :class:`Tool` is anything with a ``name``, a ``description``, and a
``run(arg)`` method returning a :class:`ToolResult`. The protocol is a
:class:`typing.Protocol`, so existing classes satisfy it structurally; a
:class:`BaseTool` convenience base is also provided.

Example tools:

* :class:`CalculatorTool` — safe arithmetic via an AST allow-list (no ``eval``).
* :class:`EchoTool` — returns its input unchanged.
* :class:`MemorySearchTool` — keyword search over a :class:`~ai_agent_framework.memory.Memory`.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai_agent_framework.memory import Memory


class ToolError(Exception):
    """Raised when a tool cannot complete its work for the given input."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of a single tool invocation."""

    output: str
    ok: bool = True


@runtime_checkable
class Tool(Protocol):
    """Structural protocol every tool must satisfy."""

    name: str
    description: str

    def run(self, arg: str) -> ToolResult:
        """Execute the tool against ``arg`` and return a :class:`ToolResult`."""
        ...


class BaseTool:
    """Optional convenience base class implementing the :class:`Tool` protocol."""

    name: str = "base"
    description: str = "Abstract base tool."

    def run(self, arg: str) -> ToolResult:  # pragma: no cover - overridden
        raise NotImplementedError


class ToolRegistry:
    """Holds the tools available to an agent, keyed by name."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises :class:`ValueError` if the name is taken."""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a tool by name. Raises :class:`KeyError` if missing."""
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Whether a tool with this name is registered."""
        return name in self._tools

    def names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)

    def catalogue(self) -> str:
        """A human/LLM-readable list of tools and their descriptions."""
        if not self._tools:
            return "  (none)"
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


class EchoTool(BaseTool):
    """Return the input unchanged — the simplest possible tool."""

    name = "echo"
    description = "Echo the input text back to the caller."

    def run(self, arg: str) -> ToolResult:
        return ToolResult(output=arg)


class MemorySearchTool(BaseTool):
    """Search the agent's memory for records related to a query."""

    name = "memory_search"
    description = "Search the agent's memory for records related to a query."

    def __init__(self, memory: Memory, *, limit: int = 3) -> None:
        self._memory = memory
        self._limit = limit

    def run(self, arg: str) -> ToolResult:
        matches = self._memory.search(arg, limit=self._limit)
        if not matches:
            return ToolResult(output=f"No memory found for: {arg}", ok=True)
        return ToolResult(output="; ".join(record.content for record in matches))


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(BaseTool):
    """Evaluate a basic arithmetic expression like ``2 * (3 + 4)`` safely."""

    name = "calculator"
    description = "Evaluate an arithmetic expression (+, -, *, /, //, %, ** and parens)."

    def run(self, arg: str) -> ToolResult:
        try:
            tree = ast.parse(arg, mode="eval")
            value = _eval(tree.body)
        except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as exc:
            return ToolResult(output=f"calculator error: {exc}", ok=False)
        if value == int(value):
            return ToolResult(output=str(int(value)))
        return ToolResult(output=str(value))


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ValueError("only numeric literals are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return unary(_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")
