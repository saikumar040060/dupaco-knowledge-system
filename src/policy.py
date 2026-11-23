"""
The "what's current" policy — the one place that decides, among several
competing FactRows for the *same* product/tier/metric, which value is true.

Written down once so it can be defended in one sentence: prefer the most
recently-crawled snapshot that has the fact at all; within that snapshot,
prefer the more authoritative page type (the /rates hub and a product's own
page, over a calculator/promo, over a blog/press mention); within a tie on
authority, prefer the table whose own "effective as of" date is newest;
within a tie on that, prefer whichever copy was fetched last.

This order is deliberate: recency of the *pull* dominates, because a page
that disappeared or changed by the latest snapshot shouldn't be outvoted by
an authoritative-looking page that's simply stale. Authority is the
secondary signal because two pages crawled on the same day can still
disagree (a live rates table vs. a static calculator). The effective date
is the tie-breaker of last resort for two equally authoritative pages (e.g.
the deposit and loan halves of the /rates page are stamped on different
days).
"""
from __future__ import annotations

from facts import AUTHORITY_RANK

EPSILON = 0.005  # percentage points; values this close are the same value


def _effective_key(fact) -> str:
    return fact.effective_date or "0000-00-00"


def filter_as_of(facts: list, as_of: str | None) -> list:
    if not as_of:
        return facts
    return [f for f in facts if f.taken_on <= as_of]


def resolve_current(facts: list) -> list:
    """Narrow a list of candidate FactRows (already filtered to one
    product+tier+metric) down to the winning one(s). Returns a list because
    a single table row can legitimately report two metrics (dividend rate
    and APY) that are both "the" current value."""
    candidates = list(facts)
    if not candidates:
        return []

    latest_snapshot = max(c.taken_on for c in candidates)
    candidates = [c for c in candidates if c.taken_on == latest_snapshot]

    best_authority = min(AUTHORITY_RANK.get(c.page_type, 3) for c in candidates)
    candidates = [c for c in candidates if AUTHORITY_RANK.get(c.page_type, 3) == best_authority]

    best_eff = max(_effective_key(c) for c in candidates)
    candidates = [c for c in candidates if _effective_key(c) == best_eff]

    # Citation preference only (every candidate left agrees on the value):
    # when a product's own page and the /rates aggregator are still tied,
    # point members at the product's own page.
    product_pages = [c for c in candidates if c.page_type == "product"]
    if product_pages:
        candidates = product_pages

    best_fetch = max(c.fetched_at for c in candidates)
    candidates = [c for c in candidates if c.fetched_at == best_fetch]

    return candidates


def resolve_entity_current(facts: list) -> list:
    """Same idea as resolve_current but for EntityFacts, which carry an
    article/effective date instead of a per-table one, and whose page_type
    authority order treats dated news/blog articles as lower authority than
    a live page — EXCEPT when the only source for a fact *is* a dated
    article (e.g. a branch-count figure that only ever appears in press-
    release boilerplate), in which case the newest article date wins."""
    candidates = list(facts)
    if not candidates:
        return []
    latest_snapshot = max(c.taken_on for c in candidates)
    candidates = [c for c in candidates if c.taken_on == latest_snapshot]

    best_authority = min(AUTHORITY_RANK.get(c.page_type, 3) for c in candidates)
    candidates = [c for c in candidates if AUTHORITY_RANK.get(c.page_type, 3) == best_authority]

    dated = [c for c in candidates if c.effective_date]
    if dated:
        best_eff = max(c.effective_date for c in dated)
        candidates = [c for c in dated if c.effective_date == best_eff]

    best_fetch = max(c.fetched_at for c in candidates)
    candidates = [c for c in candidates if c.fetched_at == best_fetch]
    return candidates


def values_differ(a: float, b: float) -> bool:
    return abs(a - b) > EPSILON


def authority_label(page_type: str) -> str:
    return {
        "rates_hub": "the official rates page",
        "product": "the product's own page",
        "branch": "the branch locator",
        "calculator": "a calculator/promo page",
        "news": "a blog or press-release page",
        "other": "a secondary page",
    }.get(page_type, page_type)
