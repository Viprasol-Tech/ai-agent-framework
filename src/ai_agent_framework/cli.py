# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Command-line interface for the AI Agent Framework.

Runs an agent with the built-in tools against a goal, using the deterministic
:class:`~ai_agent_framework.llm.FakeLLM` by default so it works with no API key.
Pass ``--llm openai`` to use a configured OpenAI-compatible backend.

Examples:
    agent-framework run "2 * (3 + 4)"
    agent-framework run "recall the summit date" --verbose
    agent-framework version
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ai_agent_framework import __version__
from ai_agent_framework.llm import FakeLLM, LLMClient
from ai_agent_framework.providers import build_agent


def _make_llm(name: str) -> LLMClient:
    if name == "fake":
        return FakeLLM()
    if name == "openai":
        from ai_agent_framework.llm import OpenAIClient

        return OpenAIClient()
    raise SystemExit(f"unknown llm backend: {name!r}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command-line parser."""
    parser = argparse.ArgumentParser(
        prog="agent-framework",
        description="Minimal LLM agent framework by Viprasol Tech (https://viprasol.com).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print the installed version.")

    run_cmd = sub.add_parser("run", help="Run the agent against a goal.")
    run_cmd.add_argument("goal", help="The objective for the agent.")
    run_cmd.add_argument(
        "--llm", default="fake", choices=("fake", "openai"), help="LLM backend to use."
    )
    run_cmd.add_argument("--verbose", action="store_true", help="Print each step.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``agent-framework`` console script."""
    args = build_parser().parse_args(argv)

    if args.command == "version":
        print(f"ai-agent-framework {__version__} - by Viprasol Tech")
        return 0

    if args.command == "run":
        agent = build_agent(_make_llm(args.llm))
        result = agent.run(args.goal)
        if args.verbose:
            for index, step in enumerate(result.steps, start=1):
                if step.tool:
                    print(f"[step {index}] {step.tool}({step.tool_input!r}) -> {step.observation}")
                else:
                    print(f"[step {index}] {step.thought}")
            print(f"[status] {result.status}")
        print(result.answer)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
