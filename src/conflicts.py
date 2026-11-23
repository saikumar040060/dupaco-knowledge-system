"""
conflicts(): every place the website contradicts itself.

One `Conflict` is emitted per (product, normalized tier/qualifiers, metric)
where two or more current-looking pages give *different* values for that exact
cell — e.g. "auto & motorcycle loans, 2024-27 model year, 60-month APR:
5.34% (calculator, effective 1/1/2026) vs 5.39% (rates page, effective
08/31/2026)". The entry lists only the disagreeing values, the pages behind
them, the winner, and a dated reason. Every entry is therefore individually
defensible — never a bundle that mixes "5.19% vs 8.54%" across unrelated tiers.

Three things are deliberately NOT conflicts, all called out in the brief:
  - jumbo vs standard, Roth vs Traditional, one balance/term/model-year tier
    vs another -> DIFFERENT facts, folded into the tier key, never compared.
  - a value that simply changed between snapshots and was replaced everywhere
    -> invisible, because we only compare within the latest snapshot.
  - an equal-authority, equal-date difference -> treated as rounding/parse
    noise; a real conflict needs a demonstrably stale loser (older effective
    date, or a less authoritative page than the winner).

Model-year tier labels drift over time ("2023-26" -> "2024-27") while meaning
the same "newest vehicles" tier, so those are aligned by their *rank* (newest
= rank 0) within a product+term on each page, not by the literal years.
Amount tiers ($832,750, $300k, ...) line up exactly across pages, so they key
on the ceiling directly.
"""
from __future__ import annotations

from collections import defaultdict

import policy
from contract import Conflict, Source
from qa import BUCKETS, _matches_bucket


def _bucket_of(fact) -> str | None:
    for name, _, _ in BUCKETS:
        if _matches_bucket(fact, name):
            return name
    return None


def _model_year_buckets(rows: list) -> dict:
    """id(row) -> canonical model-year bucket. Tier labels drift by a year or
    two across pages ("2023-26" vs "2024-27"), so cluster the distinct upper
    bounds across ALL pages for this product: bounds within one year of each
    other are the same tier ("newest"), while a 2016-and-older tier (hi 2016)
    stays separate. This replaces rank-alignment, which wrongly paired the
    newest tier on one page with the oldest tier on a page that happened to
    list fewer tiers for that term."""
    his = sorted({r.qualifiers["model_year_hi"] for r in rows
                  if r.qualifiers.get("model_year_hi") is not None})
    bucket_of_hi, bucket_id, prev = {}, -1, None
    for hi in his:
        if prev is None or hi - prev > 1:
            bucket_id += 1
        bucket_of_hi[hi] = bucket_id
        prev = hi
    return {id(r): bucket_of_hi[r.qualifiers["model_year_hi"]]
            for r in rows if r.qualifiers.get("model_year_hi") is not None}


def _tier_key(fact, my_bucket) -> tuple:
    q = fact.qualifiers
    return (
        q.get("term_months"),
        bool(q.get("is_jumbo")),
        q.get("ira_type"),
        q.get("amount_hi"),
        q.get("amount_lo") if q.get("amount_hi") is None else None,
        my_bucket,
    )


def _tier_label(fact) -> str:
    """Human-readable tier description for a conflict topic."""
    q = fact.qualifiers
    parts = []
    if q.get("model_year_lo") is not None and q.get("model_year_hi") is not None:
        lo, hi = q["model_year_lo"], q["model_year_hi"]
        parts.append(f"{lo} model year" if lo == hi else f"{lo}-{hi} model year")
    if q.get("ira_type"):
        parts.append(q["ira_type"].capitalize())
    if q.get("is_jumbo"):
        parts.append("jumbo")
    if q.get("amount_hi") is not None:
        parts.append(f"${q['amount_hi']:,.0f} or less")
    elif q.get("amount_lo") is not None:
        parts.append(f"${q['amount_lo']:,.0f}+")
    if q.get("term_months") is not None:
        t = q["term_months"]
        parts.append(f"{t // 12}-year" if t >= 12 and t % 12 == 0 and t >= 120 else f"{t}-month")
    return ", ".join(parts)


def rate_conflicts(store) -> list:
    if not store.snapshots:
        return []
    latest = store.snapshots[-1].taken_on
    # Only rate metrics can contradict each other here; fees and the
    # minimum-to-open/earn dollar columns are not rate conflicts.
    rate_metrics = {"apr", "apy", "dividend_rate", "rate"}
    latest_facts = store.facts_where(lambda f: f.metric in rate_metrics, taken_on=latest)

    by_bucket = defaultdict(list)
    for f in latest_facts:
        b = _bucket_of(f)
        if b is None:
            continue
        by_bucket[b].append(f)

    out = []
    for bucket, bucket_rows in by_bucket.items():
        year_buckets = _model_year_buckets(bucket_rows)
        by_tier = defaultdict(list)
        for r in bucket_rows:
            by_tier[(r.metric,) + _tier_key(r, year_buckets.get(id(r)))].append(r)

        # One Conflict per (product, tier, metric) cell where current pages
        # disagree and at least one losing value is demonstrably stale.
        for tier_rows in by_tier.values():
            if len({r.url for r in tier_rows}) < 2 or len({r.display() for r in tier_rows}) < 2:
                continue
            winner = policy.resolve_current(tier_rows)
            if not winner:
                continue
            win = winner[0]
            stale_losers = [r for r in tier_rows
                            if r.display() != win.display() and _is_stale_against(r, win)]
            if not stale_losers:
                continue

            # Cite the winning value from the page whose URL best matches the
            # product family — the canonical rates/product page — not an
            # incidental copy of the table embedded on, say, a home-improvement
            # page. (Same rule the answer path uses for its citations.)
            winner_rows = [r for r in tier_rows if r.display() == win.display()]
            cite = _canonical_row(winner_rows, win.family)

            values = sorted({r.display() for r in tier_rows if r.display() == win.display()
                             or r in stale_losers})
            tier = _tier_label(win)
            metric_name = {"apr": "APR", "apy": "APY", "dividend_rate": "dividend rate",
                           "rate": "rate"}.get(win.metric, win.metric)
            topic = f"{win.family}" + (f", {tier}" if tier else "") + f", {metric_name}"
            loser = max(stale_losers, key=lambda r: (r.effective_date or ""))
            reason = (
                f"{policy.authority_label(cite.page_type)} ({cite.url}) effective "
                f"{cite.effective_date or cite.taken_on} shows {cite.display()}, while "
                f"{policy.authority_label(loser.page_type)} ({loser.url}) effective "
                f"{loser.effective_date or loser.taken_on} still shows {loser.display()}; "
                f"the more authoritative / more recently effective value wins."
            )
            rows_for_sources = [cite] + winner_rows + stale_losers
            out.append(Conflict(topic=topic, values=values,
                                 sources=_dedupe_sources(rows_for_sources),
                                 resolution=win.display(), reason=reason))
    return out


def _canonical_row(rows: list, family: str):
    """Among rows sharing the winning value, the one whose URL path best matches
    the product family (then a product page over the rates hub, then shortest
    path). Picks /personal/loans/home-loans over an incidental embed."""
    import retrieval
    fam_tokens = set(retrieval.tokenize(family))

    def score(r):
        url_tokens = set(retrieval.tokenize(r.url.replace("/", " ").replace("-", " ")))
        return (len(fam_tokens & url_tokens), 1 if r.page_type == "product" else 0, -len(r.url))

    return max(rows, key=score)


def _is_stale_against(loser, winner) -> bool:
    """A losing value counts as a real stale contradiction only if its page
    is less authoritative than the winner's, or its table's effective date
    is older. Equal-authority, equal-date disagreements are treated as noise
    (rounding, parse ambiguity) and dropped."""
    from facts import AUTHORITY_RANK
    if AUTHORITY_RANK.get(loser.page_type, 3) > AUTHORITY_RANK.get(winner.page_type, 3):
        return True
    lo = loser.effective_date or ""
    wo = winner.effective_date or ""
    return bool(lo and wo and lo < wo)


def _dedupe_sources(rows: list) -> list:
    seen, out = set(), []
    for r in rows:
        if r.url not in seen:
            seen.add(r.url)
            out.append(Source(url=r.url, snapshot_id=r.snapshot_id, quote=r.quote[:200]))
    return out


def entity_conflicts(store) -> list:
    """Entity-level disagreements that coexist in the latest snapshot: the
    branch count and the Forbes recognition count both appear with several
    different values at once because old press releases stay archived and
    live. Resolve to the value carried by the most recently dated source."""
    if not store.snapshots:
        return []
    latest = store.snapshots[-1].taken_on
    by_key = defaultdict(list)
    for e in store.entities_where(taken_on=latest):
        by_key[e.key].append(e)

    out = []
    for key, topic in (("branch_count", "number of branch office locations"),
                       ("forbes_nth", "number of times Forbes has recognized Dupaco")):
        facts = by_key.get(key, [])
        values = sorted({f.value for f in facts})
        if len(values) < 2:
            continue
        winners = policy.resolve_entity_current(facts)
        if not winners:
            continue
        win = winners[0]
        sources = _dedupe_sources([type("S", (), {"url": f.url, "snapshot_id": f.snapshot_id,
                                                   "quote": f.quote})() for f in facts])
        reason = ("the most recently dated source states this value; older press releases "
                  "predate later changes and were never taken down.")
        out.append(Conflict(topic=topic, values=values, sources=sources,
                             resolution=win.value, reason=reason))
    return out


def all_conflicts(store) -> list:
    return rate_conflicts(store) + entity_conflicts(store)
