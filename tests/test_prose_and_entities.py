"""
Regression tests for the second round of fixes:
  1. prose extractor skips calls-to-action / FAQ headings
  2. structured "Min. to Open" lookup
  3. VTO reports all current year figures
  4. multi-value facts (branch phones, lobby schedules) keep every current value
  5. routing-number extraction across phrasings
These run against the real snapshots, skipped on a fresh clone until unpacked.
"""
import json
from pathlib import Path

import pytest

from contract import Page, Snapshot
import solution
from facts import ROUTING_RE
import qa

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


# ---- 1. CTA / sentence filtering (no snapshots needed) --------------------

def test_cta_lines_are_filtered():
    assert list(qa._clean_sentences("I'm ready to open my Launchpad savings account")) == []
    assert list(qa._clean_sentences("HOW TO Open a Bright Savings account")) == []
    assert list(qa._clean_sentences("Apply today for a loan")) == []
    assert list(qa._clean_sentences("Learn more about our rates")) == []


def test_short_and_question_fragments_dropped():
    assert list(qa._clean_sentences("Open now")) == []            # too short
    assert list(qa._clean_sentences("How do I open a Bright Savings account?")) == []  # FAQ heading


def test_real_sentence_kept():
    s = "This account is available exclusively to consumer members and small businesses."
    assert s in list(qa._clean_sentences(s))


# ---- 5. routing-number extraction across phrasings ------------------------

@pytest.mark.parametrize("text", [
    "Routing Number: 273974581",
    "share our routing number (273974581) and your account info",
    "Provide the payer with our ABA number (273974581).",
])
def test_routing_regex_matches_each_phrasing(text):
    m = ROUTING_RE.search(text)
    assert m and (m.group(1) or m.group(2)) == "273974581"


# ---- end-to-end behaviours ------------------------------------------------

@snapshot_test
def test_min_to_open_is_structured(ks):
    a = ks.answer("What's the minimum deposit to open a Launchpad Savings account?")
    assert "$5" in a.text and a.confident


@snapshot_test
def test_vto_reports_all_years(ks):
    a = ks.answer("How many Dupaco employees used Volunteer Time Off last year?")
    assert "294" in a.text          # expected figure
    assert "256" in a.text          # the other current year is included, not hidden


@snapshot_test
def test_lobby_hours_lists_all_distinct_schedules(ks):
    a = ks.answer("Do all Dupaco branches keep the same lobby hours?")
    assert "12:30" in a.text and "noon" in a.text


@snapshot_test
def test_branch_phone_lists_all_current_numbers(ks):
    a = ks.answer("What's the phone number for the Dupaco Learning Lab branch?")
    assert "366-8231" in a.text     # all matching branch lines, not the main line


@snapshot_test
def test_routing_cites_canonical_page(ks):
    a = ks.answer("What is Dupaco's routing number?")
    assert "273974581" in a.text
    urls = {s.url for s in a.sources}
    assert any("business/cash-management" in u for u in urls)


@snapshot_test
def test_mortgage_cites_product_page_not_incidental_embed(ks):
    a = ks.answer("Can you tell me the interest rate for a 30-year mortgage?")
    urls = {s.url for s in a.sources}
    assert any("home-loans" in u for u in urls)
    assert not any("home-improvement" in u for u in urls)


@snapshot_test
def test_15_year_mortgage_does_not_leak_30_year_rate(ks):
    a = ks.answer("What is the current rate on a 15-year fixed mortgage?")
    assert "6.75" in a.text
    assert "7.25" not in a.text     # the 30-year value must not appear


@snapshot_test
def test_bright_savings_eligibility(ks):
    a = ks.answer("Can a business open a Bright Savings account?")
    assert "consumer" in a.text.lower()


@snapshot_test
def test_mortgage_above_top_tier_refuses(ks):
    # Nothing is published above the $832,750 ceiling -> refuse, don't quote it.
    a = ks.answer("What is the interest rate on a jumbo mortgage above $832,750?")
    assert a.confident is False
    assert "7.25" not in a.text


@snapshot_test
def test_term_specific_mortgage_returns_only_that_term(ks):
    a = ks.answer("What's the current rate on a 15-year fixed mortgage?")
    assert "6.75" in a.text
    assert "7.25" not in a.text and "7.13" not in a.text  # no 20/30-year rows


@snapshot_test
def test_conflicts_have_no_dollar_minimum_noise(ks):
    # min-to-open/earn dollar columns must never appear as rate conflicts
    for c in ks.conflicts():
        assert not any(v.strip().startswith("$") for v in c.values), c.topic


@snapshot_test
def test_checking_fee_question_grounded_not_rates(ks):
    a = ks.answer("What are the fees for Dupaco's different checking account types?")
    assert "$10" in a.text and "$5,000" in a.text           # 1st Rate monthly fee + waiver
    assert "no monthly fee" in a.text.lower()                # Free Checking
    assert "%" not in a.text                                 # not APY rates
    assert any("checking-accounts" in s.url for s in a.sources)


@snapshot_test
def test_checking_rates_question_still_returns_apy(ks):
    a = ks.answer("Provide current checking account interest rates")
    assert "2.50%" in a.text                                 # rate path unaffected


@snapshot_test
def test_private_or_ceo_phone_abstains(ks):
    for q in ["What is the private phone number of Dupaco's CEO?",
              "What's the CEO's direct line?",
              "Give me the president's cell number"]:
        a = ks.answer(q)
        assert a.confident is False, q
    assert "800-373-7600" in ks.answer("What is your phone number?").text   # general still works
