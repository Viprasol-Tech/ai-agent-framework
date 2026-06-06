# Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""Tests for the Memory store and its deterministic keyword search."""

from __future__ import annotations

from ai_agent_framework.memory import Memory, MemoryRecord


def test_add_returns_record_and_grows_store() -> None:
    memory = Memory()
    record = memory.add("hello world", kind="note")
    assert isinstance(record, MemoryRecord)
    assert record.content == "hello world"
    assert record.kind == "note"
    assert len(memory) == 1


def test_search_ranks_by_overlap() -> None:
    memory = Memory()
    memory.add("the launch date is in March")
    memory.add("the budget is fixed")
    memory.add("launch checklist and launch plan")
    results = memory.search("launch")
    assert results
    assert results[0].content == "launch checklist and launch plan"


def test_search_empty_query_returns_nothing() -> None:
    memory = Memory()
    memory.add("anything")
    assert memory.search("   ") == []


def test_search_respects_limit_and_is_iterable() -> None:
    memory = Memory()
    for i in range(5):
        memory.add(f"item number {i}")
    results = memory.search("item", limit=2)
    assert len(results) == 2
    assert list(memory) == memory.records
