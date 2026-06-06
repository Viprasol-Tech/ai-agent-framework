# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""A runnable, offline demo of the AI Agent Framework.

Run it with no arguments and no API key::

    python examples/demo.py

It seeds an agent's memory, then runs three goals through the perceive/think/act
loop using the deterministic FakeLLM, printing each step.
"""

from __future__ import annotations

from ai_agent_framework import (
    Agent,
    CalculatorTool,
    EchoTool,
    FakeLLM,
    Memory,
    MemorySearchTool,
    ToolRegistry,
)


def main() -> None:
    memory = Memory()
    memory.add("The Viprasol AI summit is on March 1st.", kind="fact")
    memory.add("Viprasol Tech builds fintech and AI agent software.", kind="fact")

    agent = Agent(
        FakeLLM(),
        ToolRegistry([CalculatorTool(), EchoTool(), MemorySearchTool(memory)]),
        memory=memory,
    )

    goals = [
        "2 * (3 + 4)",
        "recall the summit date",
        "say hello to the world",
    ]

    for goal in goals:
        print(f"\n>>> goal: {goal}")
        result = agent.run(goal)
        for index, step in enumerate(result.steps, start=1):
            if step.tool:
                print(f"  step {index}: {step.tool}({step.tool_input!r}) -> {step.observation}")
        print(f"  answer: {result.answer}")


if __name__ == "__main__":
    main()
