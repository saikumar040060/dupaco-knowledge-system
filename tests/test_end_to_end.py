"""
End-to-end against the real dupaco snapshots (skipped automatically if the
snapshots haven't been unpacked yet — e.g. a fresh clone before the first
an end-to-end run with the snapshots present). Covers the refusal path, as_of filtering, determinism, and the
headline stale-trap cases.
"""
import json
from pathlib import Path

import pytest

from contract import Page, Snapshot
import solution

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "snapshots"


def _have_snapshots():
    return SNAP_DIR.is_dir() and any(
        (d / "manifest.json").exists() for d in SNAP_DIR.iterdir() if d.is_dir()
    )


pytestmark = pytest.mark.skipif(not _have_snapshots(),
                                reason="snapshots not present; add a snapshots/ corpus first")


@pytest.fixture(scope="module")
def ks():
    k = solution.build()
    dirs = sorted(d for d in SNAP_DIR.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    for d in dirs:
        m = json.loads((d / "manifest.json").read_text())
        pages = [
            Page(url=p["url"], title=p.get("title", ""),
                 markdown=(d / p["file"]).read_text(encoding="utf-8"),
                 fetched_at=p.get("fetched_at", m["taken_on"]))
            for p in m["pages"]
        ]
        k.ingest(Snapshot(m["snapshot_id"], m["taken_on"], pages))
    return k


def _nums(text):
    import re
    return set(re.findall(r"\d+\.\d+", text))


def test_auto_loan_prefers_current_rates_page_not_stale_calculator(ks):
    a = ks.answer("What's the APR for a 60-month auto loan on a 2025 model-year vehicle?")
    assert "5.39%" in a.text
    assert "5.34%" not in a.text  # the stale calculator value must never appear


def test_refuses_unlisted_product(ks):
    a = ks.answer("What is your rate on a 7/1 adjustable-rate mortgage?")
    assert a.confident is False


def test_as_of_returns_historical_value(ks):
    a = ks.answer("What was the 10-month fixed jumbo certificate special APY?", as_of="2026-05-31")
    assert "3.90%" in a.text
    assert "4.00%" not in a.text


def test_routing_number(ks):
    a = ks.answer("What is Dupaco's routing number?")
    assert "273974581" in a.text


def test_answer_is_deterministic(ks):
    q = "What are the mortgage rates right now?"
    assert _nums(ks.answer(q).text) == _nums(ks.answer(q).text)


def test_conflicts_are_precise_and_resolved(ks):
    conflicts = ks.conflicts()
    assert conflicts, "expected at least the headline conflicts"
    # Per-tier design: each entry names one (product, tier, metric) and lists
    # only the disagreeing values — never a bundle mixing unrelated tiers.
    import re
    for c in conflicts:
        nums = [float(x) for x in re.findall(r"(\d+\.\d+)", " ".join(c.values))]
        if nums:
            assert max(nums) - min(nums) <= 3.5, (c.topic, c.values)
    auto60 = [c for c in conflicts if "auto" in c.topic.lower() and "60-month" in c.topic]
    assert auto60 and auto60[0].values == ["5.34% APR", "5.39% APR"]
    assert "5.39%" in (auto60[0].resolution or "")
