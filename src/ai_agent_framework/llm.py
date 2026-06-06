# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""LLM client abstraction plus a deterministic offline implementation.

An :class:`LLMClient` turns a list of :class:`Message` objects into a single
assistant reply. Keeping this surface tiny means any backend — a local model, a
hosted API, or the deterministic :class:`FakeLLM` used in tests — can be dropped
in without touching the rest of the framework.

The agent speaks a small ReAct-style protocol. Each reply is one line, either::

    ACTION: <tool_name>(<argument>)
    FINAL: <answer>

:class:`FakeLLM` understands that protocol and needs no network or API key.
An optional :class:`OpenAIClient` (lazy import, no hard dependency) is provided
for real model access.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]

_ARITHMETIC_RE = re.compile(r"^[\d\s.+\-*/()]+$")
_RECALL_RE = re.compile(r"\b(recall|remember|earlier|previously|what did)\b", re.IGNORECASE)
_GOAL_RE = re.compile(r"Goal:\s*(.*)")


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message exchanged with an :class:`LLMClient`."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"invalid role: {self.role!r}")


class LLMClient(ABC):
    """Abstract base class for chat-style large language model clients."""

    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """Return the assistant reply for the given conversation history.

        Args:
            messages: The full conversation so far, oldest first.

        Returns:
            The assistant's reply as plain text.
        """
        raise NotImplementedError


class FakeLLM(LLMClient):
    """A deterministic LLM client that requires no external services.

    Behaviour is rule-based and fully reproducible:

    * If a tool observation is present in the latest turn, it finishes.
    * If the goal looks like arithmetic, it calls the ``calculator`` tool.
    * If the goal asks to recall something, it calls ``memory_search``.
    * Otherwise it echoes the goal via the ``echo`` tool.

    You can also pass a fixed ``script`` of replies for precise test control.
    """

    def __init__(self, script: Sequence[str] | None = None) -> None:
        """Create a FakeLLM.

        Args:
            script: Optional fixed sequence of replies. When provided, each call
                returns the next entry; once exhausted the client falls back to
                its rule-based logic.
        """
        self._script: list[str] = list(script) if script is not None else []
        self._cursor = 0

    def complete(self, messages: list[Message]) -> str:
        if self._cursor < len(self._script):
            reply = self._script[self._cursor]
            self._cursor += 1
            return reply
        return self._rule_based_reply(messages)

    @staticmethod
    def _rule_based_reply(messages: list[Message]) -> str:
        prompt = messages[-1].content if messages else ""

        # If the scratchpad already holds an observation, finish with it.
        if "OBSERVATION:" in prompt:
            last = prompt.strip().splitlines()[-1]
            observation = last.replace("OBSERVATION:", "").strip()
            return f"FINAL: {observation}"

        goal = _extract_goal(prompt)
        if not goal:
            return "FINAL: I have nothing to do."

        if _ARITHMETIC_RE.match(goal) and any(c.isdigit() for c in goal):
            return f"ACTION: calculator({goal})"

        if _RECALL_RE.search(goal):
            query = _RECALL_RE.sub("", goal).strip(" ?.!") or goal
            return f"ACTION: memory_search({query})"

        return f"ACTION: echo({goal})"


class OpenAIClient(LLMClient):
    """An optional OpenAI-compatible chat client.

    The ``openai`` package is imported lazily, so importing this module and
    running the test suite never requires the dependency. Install it with::

        pip install "ai-agent-framework[openai]"
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without dep
            raise RuntimeError(
                "The 'openai' package is required for OpenAIClient. "
                'Install it with: pip install "ai-agent-framework[openai]"'
            ) from exc
        if not self._api_key:
            raise RuntimeError("No API key provided and OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(self, messages: list[Message]) -> str:  # pragma: no cover - needs network
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""


def _extract_goal(prompt: str) -> str:
    match = _GOAL_RE.search(prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()
