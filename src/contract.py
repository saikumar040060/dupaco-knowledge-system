"""
The data model that crosses the system boundary.

A knowledge system is any object that implements the ``KnowledgeSystem``
protocol below:

    ks = build()
    for snap in snapshots (oldest -> newest):
        ks.ingest(snap)                       # Snapshot
    ks.answer(question, as_of=None)           # -> Answer
    ks.conflicts()                            # -> list[Conflict]

Only these standard-library dataclasses cross the boundary, so the internals
are free to use any approach (here: a versioned fact store + BM25 retrieval,
all pure-Python and deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    markdown: str          # page content as markdown
    fetched_at: str        # ISO-8601 timestamp of the crawl


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str       # e.g. "01_2026-05-12_baseline"
    taken_on: str          # ISO date, e.g. "2026-05-12"
    pages: list[Page]      # the full website as of this pull


@dataclass
class Source:
    url: str
    snapshot_id: Optional[str] = None
    quote: Optional[str] = None   # short supporting excerpt (optional)


@dataclass
class Answer:
    text: str                                   # the member-facing answer
    sources: list[Source] = field(default_factory=list)
    confident: bool = True                      # False = "I don't know / please contact us"
    conflict_note: Optional[str] = None         # set when sources disagree and why a value won


@dataclass
class Conflict:
    topic: str                     # e.g. "30-year fixed mortgage APR"
    values: list[str]              # the competing values
    sources: list[Source]          # where each value came from
    resolution: Optional[str] = None   # the value decided correct, if any
    reason: Optional[str] = None       # why (newer page, official rate sheet, expired promo...)


@runtime_checkable
class KnowledgeSystem(Protocol):
    def ingest(self, snapshot: Snapshot) -> None:
        """Called once per snapshot, oldest first."""

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        """Answer a question. as_of=None -> latest snapshot; as_of=date -> as the site stood then."""

    def conflicts(self) -> list[Conflict]:
        """Every place the site contradicts itself."""
