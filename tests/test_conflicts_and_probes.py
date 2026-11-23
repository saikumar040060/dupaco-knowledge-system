"""
Round-4 fixes:
  1. conflicts() emits one entry per (product, tier, metric) with only the
     disagreeing values — no cross-tier bundles.
  2. an ungrounded promotional offer ("$500 checking bonus") abstains.
  3. a "standard"/"jumbo" question returns only that variant.
"""
import json
import re
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


snapshot_test = pytest.mark.skipif(not _have_snapshots(),
                                   reason="snapshots not present; add a snapshots/ corpus first")


@pytest.fixture(scope="module")
def ks():
    k = solution.build()
    for d in sorted(x for x in SNAP_DIR.iterdir() if x.is_dir() and (x / "manifest.json").exists()):
        m = json.loads((d / "manifest.json").read_text())
        pages = [Page(url=p["url"], title=p.get("title", ""),
                      markdown=(d / p["file"]).read_text(encoding="utf-8"),
                      fetched_at=p.get("fetched_at", m["taken_on"])) for p in m["pages"]]
        k.ingest(Snapshot(m["snapshot_id"], m["taken_on"], pages))
    return k


# ---- 1. conflicts structure ----------------------------------------------

@snapshot_test
def test_conflicts_are_per_tier_not_bundles(ks):
    cs = ks.conflicts()
    for c in cs:
        nums = [float(x) for x in re.findall(r"(\d+\.\d+)", " ".join(c.values))]
        # no entry spans wildly different tiers (the old 5.19% vs 8.54% bug)
        if nums:
            assert max(nums) - min(nums) <= 3.5, (c.topic, c.values)
        # a rate entry lists exactly the handful of disagreeing values, not a
        # whole table
        assert len(c.values) <= 5, (c.topic, c.values)


@snapshot_test
def test_flagship_auto_conflict_is_specific(ks):
    cs = ks.conflicts()
    auto60 = [c for c in cs if "auto" in c.topic.lower() and "60-month" in c.topic
              and "2024-2027" in c.topic]
    assert len(auto60) == 1
    assert auto60[0].values == ["5.34% APR", "5.39% APR"]
    assert "5.39%" in auto60[0].resolution
    assert "effective" in (auto60[0].reason or "").lower()


@snapshot_test
def test_no_cross_model_year_conflict(ks):
    # The 48-month cell must not pair the newest tier with the 2016 tier.
    for c in ks.conflicts():
        if "48-month" in c.topic and "auto" in c.topic.lower():
            nums = sorted(float(x) for x in re.findall(r"(\d+\.\d+)", " ".join(c.values)))
            assert nums[-1] - nums[0] <= 1.0, (c.topic, c.values)


@snapshot_test
def test_entity_conflicts_still_present(ks):
    topics = " ".join(c.topic.lower() for c in ks.conflicts())
    assert "branch office" in topics
    assert "forbes" in topics


# ---- 2. phantom offer abstains -------------------------------------------

@snapshot_test
@pytest.mark.parametrize("q", [
    "Do you offer a student checking account bonus of $500?",
    "Is there a $1,000 sign-up bonus for a new certificate?",
])
def test_ungrounded_offer_abstains(ks, q):
    a = ks.answer(q)
    assert a.confident is False
    assert not re.search(r"\d+\.\d+%", a.text)   # no rate figures smuggled in


# ---- 3. standard vs jumbo -------------------------------------------------

@snapshot_test
def test_standard_certificate_excludes_jumbo(ks):
    a = ks.answer("What is the 10-month standard certificate rate?")
    assert "3.90%" in a.text
    assert "4.00%" not in a.text            # the jumbo value must not appear


@snapshot_test
def test_jumbo_certificate_excludes_standard(ks):
    a = ks.answer("What is the 10-month jumbo certificate rate?")
    assert "4.00%" in a.text
    assert "3.90%" not in a.text


@snapshot_test
def test_unspecified_certificate_shows_both(ks):
    a = ks.answer("What is the 10-month certificate rate?")
    assert "3.90%" in a.text and "4.00%" in a.text


# ---- round-5: prose fallback distinctive-term gate ------------------------

@snapshot_test
def test_abstains_on_unknown_product_from_blog_only(ks):
    # "crypto debit card" isn't a product; a blog that merely discusses crypto
    # must not be read as the credit union offering one.
    a = ks.answer("Do you have a crypto debit card?")
    assert a.confident is False
    assert not re.search(r"\d+\.\d+%", a.text)


@snapshot_test
def test_known_product_attribute_still_answered_from_prose(ks):
    # A fact about a real product may come from prose: Rewards Visa has no fee.
    a = ks.answer("Does the Rewards Visa charge an annual fee?")
    assert "no annual fee" in a.text.lower()


@snapshot_test
def test_structured_min_to_open_not_affected_by_prose_gate(ks):
    a = ks.answer("How much to open a Launchpad savings account?")
    assert "$5" in a.text and a.confident


# ---- round-5: conflicts cite the authoritative page ----------------------

@snapshot_test
def test_mortgage_conflict_cites_home_loans_not_home_improvement(ks):
    mort = [c for c in ks.conflicts() if "home loans" in c.topic and "100,000" in c.topic]
    assert mort
    for c in mort:
        assert "home-improvement" not in (c.reason or "")
        assert any("personal/loans/home-loans" in s.url for s in c.sources)


@snapshot_test
def test_conflict_values_are_present_in_latest_snapshot(ks):
    # Every value in a conflict must still appear on a latest-snapshot page;
    # values only replaced over time are not emitted.
    latest = ks.store.snapshots[-1]
    blob = " ".join(md for _, _, _, md in latest.pages)
    import re as _re
    for c in ks.conflicts():
        for v in c.values:
            num = _re.search(r"[\d,]+\.\d+|\d{2,}", v)
            if num:
                assert num.group(0) in blob or num.group(0).replace(",", "") in blob.replace(",", ""), (c.topic, v)
