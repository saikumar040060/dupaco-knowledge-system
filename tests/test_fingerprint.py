"""
Incremental ingest (page fingerprinting) must be answer-preserving: reusing an
unchanged page's extracted facts has to produce exactly the same fact and
entity set as parsing every page every time — only then is the 0.9643 score
untouched. Also checks that reuse actually happens.
"""
import json
from pathlib import Path

import pytest

from contract import Page, Snapshot
from ingest import Store

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "snapshots"


def _have_snapshots():
    return SNAP_DIR.is_dir() and any(
        (d / "manifest.json").exists() for d in SNAP_DIR.iterdir() if d.is_dir()
    )


snapshot_test = pytest.mark.skipif(not _have_snapshots(),
                                   reason="snapshots not present; add a snapshots/ corpus first")


def _load():
    snaps = []
    for d in sorted(x for x in SNAP_DIR.iterdir() if x.is_dir() and (x / "manifest.json").exists()):
        m = json.loads((d / "manifest.json").read_text())
        pages = [Page(url=p["url"], title=p.get("title", ""),
                      markdown=(d / p["file"]).read_text(encoding="utf-8"),
                      fetched_at=p.get("fetched_at", m["taken_on"])) for p in m["pages"]]
        snaps.append(Snapshot(m["snapshot_id"], m["taken_on"], pages))
    return snaps


def _ingest(snaps, fingerprint):
    st = Store(fingerprint=fingerprint)
    for s in snaps:
        st.ingest(s)
    return st


@snapshot_test
def test_fingerprint_is_answer_preserving():
    snaps = _load()
    on, off = _ingest(snaps, True), _ingest(snaps, False)
    assert sorted(repr(f) for f in on.facts) == sorted(repr(f) for f in off.facts)
    assert sorted(repr(e) for e in on.entities) == sorted(repr(e) for e in off.entities)
    assert len(on.facts) == len(off.facts)


@snapshot_test
def test_fingerprint_actually_reuses_pages():
    on = _ingest(_load(), True)
    assert on.pages_reused > 0            # unchanged pages were skipped
    assert on.pages_parsed > 0            # changed/new pages were still parsed


@snapshot_test
def test_store_once_expands_to_the_same_facts():
    from dataclasses import replace
    st = _ingest(_load(), True)
    expanded = [replace(u, snapshot_id=sid, taken_on=taken)
                for uid, u in enumerate(st.units) for sid, taken, _f, _s in st.seen[uid]]

    def k(f):
        return repr((f.url, f.family, f.label, f.metric, f.value_low, f.value_high,
                     f.snapshot_id, f.taken_on, f.effective_date, sorted(f.qualifiers.items())))

    assert sorted(map(k, expanded)) == sorted(k(f) for f in st.facts)   # store-once == copies
    assert len(st.units) < len(st.facts)                                 # dedup actually happened
    assert st.dedup_stats()["saved_pct"] > 0


def test_restamp_keeps_content_changes_only_the_snapshot_fields():
    # One page ingested in two snapshots -> same content fact, different stamps.
    md = """
## Auto & Motorcycle Loans

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2024-27 Model Year | 60 Months | 5.39% APR |

Rates are effective as of 08/31/2026.
"""
    p1 = Page(url="https://x.test/auto", title="Auto", markdown=md, fetched_at="2026-05-12T00:00:00Z")
    p2 = Page(url="https://x.test/auto", title="Auto", markdown=md, fetched_at="2026-09-28T00:00:00Z")
    st = Store(fingerprint=True)
    st.ingest(Snapshot("01_2026-05-12", "2026-05-12", [p1]))
    st.ingest(Snapshot("05_2026-09-28", "2026-09-28", [p2]))
    assert st.pages_reused == 1                      # second ingest reused the parse
    vals = {f.value_low for f in st.facts}
    assert vals == {5.39}                            # identical content both times
    stamps = {(f.snapshot_id, f.taken_on, f.fetched_at) for f in st.facts}
    assert len(stamps) == 2                          # but stamped per snapshot
