# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""The agent control loop: ``perceive -> think -> act`` (ReAct-style).

An :class:`Agent` wires together an :class:`~ai_agent_framework.llm.LLMClient`, a
:class:`~ai_agent_framework.memory.Memory`, and a
:class:`~ai_agent_framework.tools.ToolRegistry`, then runs a bounded loop:

#. **perceive** — the goal and any prior observations form the scratchpad prompt.
#. **think** — the LLM is asked for its next move.
#. **act** — the reply is parsed. ``ACTION: <tool>(<arg>)`` runs a tool and feeds
   the observation back; ``FINAL: <answer>`` ends the run.

The loop is deterministic when paired with
:class:`~ai_agent_framework.llm.FakeLLM`, which is what the test suite uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_agent_framework.llm import LLMClient, Message
from ai_agent_framework.memory import Memory
from ai_agent_framework.tools import Tool, ToolError, ToolRegistry

_ACTION_RE = re.compile(r"^\s*ACTION:\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)
_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.*)$", re.DOTALL)


@dataclass(slots=True)
class Step:
    """One reasoning/acting step in the agent's trace."""

    thought: str
    tool: str | None = None
    tool_input: str | None = None
    observation: str | None = None


@dataclass(slots=True)
class AgentResult:
    """The outcome of an agent run."""

    answer: str
    status: str  # "finished" or "max_steps"
    steps: list[Step] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        """Whether the agent finished before hitting ``max_steps``."""
        return self.status == "finished"


class Agent:
    """A minimal, tool-using LLM agent driven by an :class:`LLMClient`."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | list[Tool] | None = None,
        *,
        memory: Memory | None = None,
        max_steps: int = 6,
    ) -> None:
        """Create an agent.

        Args:
            llm: The language model client driving the loop.
            tools: A :class:`ToolRegistry` or a list of tools to register.
            memory: Optional shared memory store; created if omitted.
            max_steps: Hard cap on loop iterations to guarantee termination.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.memory = memory if memory is not None else Memory()
        self.max_steps = max_steps
        if isinstance(tools, ToolRegistry):
            self.tools = tools
        else:
            self.tools = ToolRegistry(tools or [])

    def _prompt(self, goal: str, scratchpad: str) -> str:
        return (
            "You are a tool-using agent. Use the available tools to reach the goal.\n"
            f"Tools:\n{self.tools.catalogue()}\n\n"
            "Reply with exactly ONE line, either:\n"
            "  ACTION: tool_name(argument)\n"
            "  FINAL: your answer\n\n"
            f"Goal: {goal}\n"
            f"Scratchpad:\n{scratchpad}\n"
        )

    def run(self, goal: str) -> AgentResult:
        """Run the loop until a FINAL answer or ``max_steps`` is reached."""
        self.memory.add(goal, kind="goal")
        scratchpad = ""
        steps: list[Step] = []

        for _ in range(self.max_steps):
            prompt = self._prompt(goal, scratchpad)  # perceive
            reply = self.llm.complete([Message("user", prompt)]).strip()  # think

            final = _FINAL_RE.match(reply)
            if final:
                answer = final.group(1).strip()
                steps.append(Step(thought=reply))
                self.memory.add(answer, kind="answer")
                return AgentResult(answer=answer, status="finished", steps=steps)

            action = _ACTION_RE.match(reply)
            if not action:
                observation = "ERROR: reply must be 'ACTION: tool(arg)' or 'FINAL: answer'"
                steps.append(Step(thought=reply, observation=observation))
            else:
                tool_name, arg = action.group(1), action.group(2).strip()
                observation = self._act(tool_name, arg)  # act
                steps.append(
                    Step(
                        thought=reply,
                        tool=tool_name,
                        tool_input=arg,
                        observation=observation,
                    )
                )
            self.memory.add(observation, kind="observation")
            scratchpad += f"{reply}\nOBSERVATION: {observation}\n"

        return AgentResult(answer="", status="max_steps", steps=steps)

    def _act(self, tool_name: str, arg: str) -> str:
        if not self.tools.has(tool_name):
            return f"ERROR: unknown tool '{tool_name}'"
        try:
            result = self.tools.get(tool_name).run(arg)
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # surface tool errors back to the LLM
            return f"ERROR: tool raised {type(exc).__name__}: {exc}"
        return result.output
