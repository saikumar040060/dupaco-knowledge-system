"""
Orchestrates ingest: for each snapshot (processed oldest -> newest, as the
contract promises), classify every page, pull out FactRows and EntityFacts,
and build a BM25 index over its pages. Everything heavy happens here, at
ingest time, which is off the latency-sensitive answer path — so
`answer()` only ever does cheap list filtering and index lookups.

History is never discarded: facts/entities/indexes are kept per snapshot.

Incremental ingest (the production pattern, in miniature)
---------------------------------------------------------
Most pages don't change between crawls. Each page is fingerprinted with a
content hash; when the same (url, title, page_type, hash) is seen again in a
later snapshot, its already-extracted facts are reused and simply re-stamped
with the new snapshot's id/date, instead of re-parsing the markdown. The fact
set produced is byte-for-byte identical to parsing every page every time
(FactRow/EntityFact content depends only on the page; only the three snapshot
stamps differ), so answers and scores are unchanged — see
tests/test_fingerprint.py. At real scale this is what lets you skip the ~95%
of pages that didn't change; here it just makes ingest leaner and is the same
incremental-ingestion pattern a production crawler uses.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from facts import classify_page, entity_facts_from_page, rows_from_page
from retrieval import BM25Index


@dataclass
class SnapshotStore:
    snapshot_id: str
    taken_on: str
    index: BM25Index = field(default_factory=BM25Index)
    pages: list = field(default_factory=list)  # (url, title, page_type, markdown)


class Store:
    def __init__(self, fingerprint: bool = True):
        self.snapshots: list[SnapshotStore] = []   # kept in ingest order == oldest->newest
        self._seq = 0
        # Incremental-ingest bookkeeping.
        self.fingerprint = fingerprint
        self._parse_cache: dict = {}               # (url, title, page_type, sha1) -> (facts, entities) templates
        self.pages_parsed = 0                      # pages actually re-parsed this run
        self.pages_reused = 0                      # pages skipped because content was unchanged
        # Store-once storage (the production shape, and the ONLY copy kept).
        # A fact's identity is its content; `units` holds each distinct fact
        # once and `seen[uid]` records every (snapshot, date, fetched_at, seq)
        # it appeared in — the in-memory form of a versioned row. The answer
        # path reads facts through `facts_where()` / the `facts` property,
        # which expands only the needed units back to per-snapshot rows in
        # ingest order, so answers are identical to keeping copies — but no
        # duplicates are stored. `units`+`seen` are exactly what a production
        # store (Postgres valid_from/valid_to) would keep.
        self.units: list = []                      # distinct FactRow content (stampless)
        self.unit_index: dict = {}                 # content key -> unit id
        self.seen: list = []                       # unit id -> [(snapshot_id, taken_on), ...]
        self.entity_units: list = []               # distinct EntityFact content, stored once
        self.entity_index: dict = {}
        self.entity_seen: list = []

    @staticmethod
    def _fact_key(f):
        return (f.url, f.title, f.page_type, f.family, f.label, f.metric,
                f.value_low, f.value_high, f.raw_low, f.raw_high,
                f.effective_date, f.quote, tuple(sorted(f.qualifiers.items())))

    def _intern_fact(self, stampless_fact, snapshot_id, taken_on):
        """Store each distinct fact once; record which snapshot saw it."""
        k = self._fact_key(stampless_fact)
        uid = self.unit_index.get(k)
        if uid is None:
            uid = len(self.units)
            self.unit_index[k] = uid
            self.units.append(stampless_fact)
            self.seen.append([])
        self.seen[uid].append((snapshot_id, taken_on, stampless_fact.fetched_at, self._seq))
        self._seq += 1
        return uid

    def _intern_entity(self, e, snapshot_id, taken_on):
        k = (e.category, e.key, e.value, e.url, e.title, e.page_type, e.effective_date, e.quote)
        uid = self.entity_index.get(k)
        if uid is None:
            uid = len(self.entity_units)
            self.entity_index[k] = uid
            self.entity_units.append(e)
            self.entity_seen.append([])
        self.entity_seen[uid].append((snapshot_id, taken_on, e.fetched_at, self._seq))
        self._seq += 1
        return uid

    @staticmethod
    def _expand(units, seen, pred, taken_on):
        """Rebuild per-snapshot facts from the store-once (units + seen) form,
        on demand. Only the units matching `pred` (and, if given, `taken_on`)
        are expanded, and results come back in original ingest order (`seq`)."""
        out = []
        for uid, u in enumerate(units):
            if not pred(u):
                continue
            for sid, taken, fetched, seq in seen[uid]:
                if taken_on is None or taken == taken_on:
                    out.append((seq, replace(u, snapshot_id=sid, taken_on=taken, fetched_at=fetched)))
        out.sort(key=lambda x: x[0])
        return [f for _, f in out]

    def facts_where(self, pred=lambda f: True, taken_on=None) -> list:
        return self._expand(self.units, self.seen, pred, taken_on)

    def entities_where(self, pred=lambda e: True, taken_on=None) -> list:
        return self._expand(self.entity_units, self.entity_seen, pred, taken_on)

    @property
    def facts(self) -> list:
        return self.facts_where()

    @property
    def entities(self) -> list:
        return self.entities_where()

    def dedup_stats(self) -> dict:
        """What store-once buys: distinct facts vs. per-snapshot copies."""
        copies = sum(len(s) for s in self.seen)
        once = len(self.units)
        return {"copies": copies, "stored_once": once,
                "entity_copies": sum(len(s) for s in self.entity_seen),
                "entity_stored_once": len(self.entity_units),
                "saved_pct": round(100 * (1 - once / copies), 1) if copies else 0.0}

    def _facts_for_page(self, page, page_type: str):
        """Extract (or reuse) this page's facts and entities. The cache key is
        the page's content hash, so a reused result is identical to a fresh
        parse except for the snapshot stamps, which we re-apply with
        `dataclasses.replace`."""
        key = None
        if self.fingerprint:
            digest = hashlib.sha1(page.markdown.encode("utf-8")).hexdigest()
            key = (page.url, page.title, page_type, digest)
            cached = self._parse_cache.get(key)
            if cached is not None:
                self.pages_reused += 1
                facts = [replace(f) for f in cached[0]]
                ents = [replace(e) for e in cached[1]]
                return self._restamp(facts, ents, page)

        self.pages_parsed += 1
        facts = rows_from_page(page.url, page.title, page.markdown, page_type,
                               "", "", "")                      # stampless template
        ents = entity_facts_from_page(page.url, page.title, page.markdown, page_type,
                                      "", "", "")
        if key is not None:
            self._parse_cache[key] = ([replace(f) for f in facts], [replace(e) for e in ents])
        return self._restamp(facts, ents, page)

    @staticmethod
    def _restamp(facts, ents, page):
        """Stamp templates with this page's snapshot identity. Set on the owning
        snapshot by ingest() just after (snapshot_id/taken_on) — here we only
        know the page, so fetched_at is applied now and the snapshot fields are
        filled by ingest()."""
        facts = [replace(f, fetched_at=page.fetched_at) for f in facts]
        ents = [replace(e, fetched_at=page.fetched_at) for e in ents]
        return facts, ents

    def ingest(self, snapshot) -> None:
        snap = SnapshotStore(snapshot_id=snapshot.snapshot_id, taken_on=snapshot.taken_on)
        for page in snapshot.pages:
            page_type = classify_page(page.url, page.title)
            snap.index.add_page(page.url, page.title, page_type, page.markdown)
            snap.pages.append((page.url, page.title, page_type, page.markdown))
            facts, ents = self._facts_for_page(page, page_type)
            for f in facts:
                self._intern_fact(f, snapshot.snapshot_id, snapshot.taken_on)  # store once, no copies
            for e in ents:
                self._intern_entity(e, snapshot.snapshot_id, snapshot.taken_on)
        snap.index.finalize()
        self.snapshots.append(snap)

    def snapshot_for(self, as_of: str | None):
        """The snapshot that represents the site at `as_of` (or latest)."""
        if not self.snapshots:
            return None
        if not as_of:
            return self.snapshots[-1]
        eligible = [s for s in self.snapshots if s.taken_on <= as_of]
        return eligible[-1] if eligible else None

    def latest_taken_on(self) -> str | None:
        return self.snapshots[-1].taken_on if self.snapshots else None
