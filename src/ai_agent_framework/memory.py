"""A lightweight, in-memory store for agent facts and observations.

:class:`Memory` keeps an ordered list of :class:`MemoryRecord` entries and offers
a deterministic keyword search (used by the ``memory_search`` example tool). It is
intentionally dependency-free; swap it for a vector database in production.

Part of AI Agent Framework by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A single stored item with an optional ``kind`` tag."""

    content: str
    kind: str = "note"


@dataclass(slots=True)
class Memory:
    """An ordered collection of :class:`MemoryRecord` entries."""

    records: list[MemoryRecord] = field(default_factory=list)

    def add(self, content: str, kind: str = "note") -> MemoryRecord:
        """Store ``content`` and return the created record."""
        record = MemoryRecord(content=content, kind=kind)
        self.records.append(record)
        return record

    def search(self, query: str, *, limit: int = 5) -> list[MemoryRecord]:
        """Return records ranked by overlap with ``query`` (most relevant first).

        Scoring counts how many distinct query words appear in each record, with
        a small bonus for exact substring matches. Ties keep insertion order, so
        results are fully deterministic.
        """
        terms = {t for t in _tokenize(query) if t}
        if not terms:
            return []

        scored: list[tuple[int, int, MemoryRecord]] = []
        for index, record in enumerate(self.records):
            tokens = _tokenize(record.content)
            # Reward every occurrence of a query term, so a record that repeats
            # a term ranks above one that mentions it only once.
            overlap = sum(1 for token in tokens if token in terms)
            if overlap == 0 and query.lower() not in record.content.lower():
                continue
            bonus = 1 if query.lower() in record.content.lower() else 0
            scored.append((overlap + bonus, -index, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MemoryRecord]:
        return iter(self.records)


def _tokenize(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()
