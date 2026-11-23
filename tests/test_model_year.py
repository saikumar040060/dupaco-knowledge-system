"""
Model-year parsing for auto-loan questions: a 4-digit year next to a vehicle
word (or "a/my/new/used <year>") selects exactly one tier, regardless of the
exact phrasing, and a year-less question still lists every tier.
"""
import json
import re
from pathlib import Path

import pytest

from contract import Page, Snapshot
import solution
from qa import MODEL_YEAR_Q_RE, parse_question

ROOT = Path(__file__).resolve().parents[1]
SNAP_DIR = ROOT / "snapshots"


def _have_snapshots():
    return SNAP_DIR.is_dir() and any(
        (d / "manifest.json").exists() for d in SNAP_DIR.iterdir() if d.is_dir()
    )


snapshot_test = pytest.mark.skipif(not _have_snapshots(),
                                   reason="snapshots not present; add a snapshots/ corpus first")


# ---- regex-level parsing (no snapshots needed) ----------------------------

@pytest.mark.parametrize("q,year", [
    ("60 months, 2025 model", 2025),
    ("a 2025 car, 60 month rate", 2025),
    ("used 2019 car, 60 month loan rate", 2019),
    ("2022 vehicle, 60 months", 2022),
    ("my 2023 Honda auto loan", 2023),
    ("2024 model-year vehicle", 2024),
])
def test_model_year_is_parsed(q, year):
    assert parse_question(q).model_year == year


@pytest.mark.parametrize("q", [
    "what is a 2,500 minimum balance account",   # loan amount, not a model year
    "60 month auto loan rate",                    # no year at all
    "rate as of May 2026",                        # a date, not a vehicle year
])
def test_non_model_year_not_misread(q):
    assert parse_question(q).model_year is None


def test_brand_new_maps_to_newest_tier():
    # "brand new car" has no explicit year; it should resolve to the newest tier.
    assert parse_question("brand new car, 60 month auto rate").model_year == 2026


# ---- end-to-end: one tier each -------------------------------------------

def _nums(text):
    return re.findall(r"\d+\.\d+%", text)


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


MODEL_YEAR_CASES = [
    ("60 months, 2025 model", "5.39%"),
    ("a 2025 car, 60 month rate", "5.39%"),
    ("used 2019 car, 60 month loan rate", "6.34%"),
    ("2022 vehicle, 60 months", "5.64%"),
    ("brand new car, 60 month auto rate", "5.39%"),
]


@snapshot_test
@pytest.mark.parametrize("q,expected", MODEL_YEAR_CASES)
def test_each_model_year_returns_one_value(ks, q, expected):
    assert _nums(ks.answer(q).text) == [expected]


@snapshot_test
def test_year_less_question_unchanged(ks):
    # No year -> still the full multi-tier answer for 60-month auto.
    nums = _nums(ks.answer("60 month auto loan rate").text)
    assert "5.39%" in nums and len(nums) > 1


@snapshot_test
def test_same_year_phrasings_are_consistent(ks):
    # All 2025 phrasings must yield the identical set of numbers (consistency).
    variants = [
        "60 months, 2025 model",
        "a 2025 car, 60 month rate",
        "2025 vehicle, 60 month auto loan rate",
        "my 2025 truck, 60 month rate",
    ]
    sets = {frozenset(_nums(ks.answer(v).text)) for v in variants}
    assert len(sets) == 1


@snapshot_test
def test_term_phrasings_and_bare_year_agree(ks):
    # "60-month ... 2025", "5-year ... 2025", "five year ... 2025" are the same.
    import re as _re
    def nums(t): return _re.findall(r"\d+\.\d+%", t)
    a = nums(ks.answer("60-month auto loan, 2025?").text)
    b = nums(ks.answer("5-year auto loan, 2025?").text)
    c = nums(ks.answer("five year auto loan, 2025?").text)
    assert a == b == c == ["5.39%"]
