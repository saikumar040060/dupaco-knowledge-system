"""The written-down 'what's current' precedence rules."""
from facts import FactRow
import policy


def _fact(value, page_type, snapshot, effective, fetched="2026-01-01T00:00:00Z", url="u"):
    return FactRow(
        family="auto & motorcycle loans", label="2024-27 Model Year", metric="apr",
        value_low=value, value_high=None, qualifiers={"term_months": 60},
        url=url, title="", page_type=page_type, snapshot_id=snapshot,
        taken_on=snapshot, fetched_at=fetched, effective_date=effective, raw_low=str(value),
    )


def test_latest_snapshot_wins_over_authority():
    # An authoritative page from an OLD snapshot must lose to any page in the
    # newest snapshot — recency of the pull dominates.
    old_authoritative = _fact(5.00, "rates_hub", "2026-05-12", "2026-05-01")
    new_calculator = _fact(5.39, "calculator", "2026-09-28", "2026-01-01")
    winners = policy.resolve_current([old_authoritative, new_calculator])
    assert [w.value_low for w in winners] == [5.39]


def test_authority_breaks_ties_within_a_snapshot():
    # Same snapshot: the rates hub beats the stale calculator.
    hub = _fact(5.39, "rates_hub", "2026-09-28", "2026-08-31", url="hub")
    calc = _fact(5.34, "calculator", "2026-09-28", "2026-01-01", url="calc")
    winners = policy.resolve_current([hub, calc])
    assert [w.value_low for w in winners] == [5.39]


def test_effective_date_breaks_authority_ties():
    older = _fact(7.00, "rates_hub", "2026-09-28", "2026-06-01", url="a")
    newer = _fact(7.25, "rates_hub", "2026-09-28", "2026-09-25", url="b")
    winners = policy.resolve_current([older, newer])
    assert [w.value_low for w in winners] == [7.25]


def test_product_page_preferred_for_citation_on_full_tie():
    hub = _fact(5.39, "rates_hub", "2026-09-28", "2026-08-31", url="hub")
    product = _fact(5.39, "product", "2026-09-28", "2026-08-31", url="product")
    winners = policy.resolve_current([hub, product])
    assert len(winners) == 1 and winners[0].url == "product"


def test_as_of_filters_future_snapshots():
    facts = [
        _fact(2.95, "rates_hub", "2026-01-05", "2026-01-01"),
        _fact(3.50, "rates_hub", "2026-03-01", "2026-03-01"),
    ]
    as_of_jan = policy.filter_as_of(facts, "2026-01-31")
    assert [f.value_low for f in as_of_jan] == [2.95]
    winners = policy.resolve_current(as_of_jan)
    assert winners[0].value_low == 2.95
