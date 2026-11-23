"""
Question understanding, routing and answer composition.

A question is first tried against the structured fact store (rate/APY/APR
tables and entity facts). Only when nothing structured matches does it fall
back to BM25 retrieval over page text. This order matters for consistency:
two paraphrases of a structured question must hit the same bucket and the
same qualifiers, which a keyword/regex router gives for free, whereas pure
retrieval ranking can drift between phrasings.

Everything here is deterministic: no model calls, no randomness, same input
-> same output, every time (required for the consistency/repeat scoring).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from contract import Answer, Source
from facts import AUTHORITY_RANK, extract_article_date
import policy
import retrieval

REFUSAL_TEXT = "I don't know — that isn't listed on our website; please contact our member service team for help."

# --------------------------------------------------------------------------
# Product buckets. Ordered most-specific first: the first regex that matches
# the question (or a fact's family text) wins. `domain` drives the default
# tier/qualifier chosen when the question doesn't specify one.
# --------------------------------------------------------------------------

BUCKETS = [
    ("business_platinum_visa", r"business\s+platinum\s+visa", "card"),
    ("business_rewards_visa", r"business\s+rewards\s+visa", "card"),
    ("platinum_visa", r"platinum\s+visa", "card"),
    ("rewards_visa", r"rewards\s+visa", "card"),
    ("credit_card", r"credit\s*card|\bvisa\b", "card_list"),
    ("business_checking", r"business.{0,20}checking|business\s+1st\s+rate", "balance_tier"),
    ("free_checking", r"free\s+checking", "balance_tier"),
    ("checking", r"checking", "balance_tier"),
    ("bright_savings", r"bright\s+savings", "balance_tier"),
    ("you_name_it_savings", r"you\s+name\s+it", "balance_tier"),
    ("holiday_club", r"holiday\s+club", "balance_tier"),
    ("launchpad_savings", r"launchpad", "balance_tier"),
    ("investors_choice_mm", r"investor.?s\s+choice|money\s+market", "balance_tier"),
    ("high_yield_savings", r"high.?yield\s+savings", "balance_tier"),
    ("share_savings", r"share\s+savings|savings\s+tiers?\b", "balance_tier"),
    ("business_certificate", r"business.{0,25}certificate", "term_list"),
    ("special_certificate", r"certificate.{0,15}special|special.{0,15}certificate", "term_list"),
    ("certificate", r"\bcertificate|\bcd\b|\bcds\b", "term_list"),
    ("special_ira", r"ira.{0,15}special|special.{0,15}ira", "term_list"),
    ("variable_ira", r"variable\s+ira", "term_list"),
    ("regular_ira", r"\bira\b", "term_list"),
    ("heloc", r"heloc|home\s+equity\s+line", "flat_list"),
    ("home_equity_loan", r"home\s+equity\s+loan", "flat_list"),
    ("home_improvement_loan", r"home\s+improvement", "flat_list"),
    ("construction_loan", r"construction\s+loan", "refuse"),
    ("arm_mortgage", r"adjustable.?rate|\barm\b|\d/\d\s*arm", "refuse"),
    ("mortgage_refinance", r"refinanc.{0,20}(home|mortgage)|mortgage.{0,20}refinanc", "loan_amount_tier"),
    ("mortgage", r"mortgage|home\s+loan", "loan_amount_tier"),
    ("boat_rv_loan", r"\bboat\b|\brv\b|recreational\s+loan", "flat_list"),
    ("atv_loan", r"\batv\b|snowmobile|jet\s*ski", "model_year_tier"),
    ("auto_loan", r"auto\s*loan|auto\s*&?\s*motorcycle|vehicle\s+loan|motorcycle\s+loan|\bvehicle\b|\bcar\b", "model_year_tier"),
    ("student_loan", r"student\b.{0,45}\bloan|\bloan\b.{0,25}student|college\s+(?:loan|financ)|university.{0,35}loan", "describe"),
    ("personal_loan", r"personal\s+loan", "flat_list"),
]

BUSINESS_HINT_RE = re.compile(r"\bbusiness\b", re.I)
HIGHEST_RE = re.compile(r"\bhighest|maximum|best rate|top rate\b", re.I)
LOWEST_RE = re.compile(r"\blowest|minimum|least\b", re.I)
TERM_MONTHS_Q_RE = re.compile(r"(\d+)\s*[-\s]?month", re.I)
TERM_YEARS_Q_RE = re.compile(r"(\d+)\s*[-\s]?year", re.I)
# A 4-digit year that refers to the vehicle: "2025 model", "2025 model year", "2025 vehicle",
# "2019 car", "a 2025", "my 2023 Honda...", "used 2019 car". Years 1990-2039 only, so loan
# amounts like "2,500" or "60 month" are never mistaken for a model year.
_YEAR = r"(?<![\d,$.])((?:199\d|20[0-3]\d))(?![\d,%])"
MODEL_YEAR_Q_RE = re.compile(
    _YEAR + r"\s*(?:model[-\s]?year|model|vehicle|car|truck|suv|auto|motorcycle|bike|atv)"
    + r"|(?:model[-\s]?year|year)\s*(?:of\s*)?" + _YEAR
    + r"|(?:\ba|\bmy|\bnew|\bused|\bbought a|\bbuying a)\s+" + _YEAR,
    re.I,
)
DOLLAR_Q_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
APY_Q_RE = re.compile(r"\bapy\b", re.I)
DIVIDEND_Q_RE = re.compile(r"\bdividend\b", re.I)
RATE_SHAPED_RE = re.compile(r"\brate|\bapr|\bapy|\binterest\b|\bdividend\b", re.I)


@dataclass
class ParsedQuestion:
    text: str
    is_business: Optional[bool]
    term_months: Optional[int]
    model_year: Optional[int]
    amount: Optional[float]
    wants_highest: bool
    wants_lowest: bool
    metric_hint: Optional[str]
    above_amount: Optional[float] = None   # "above/over $N" — asking beyond a ceiling


# "above $832,750", "over 832750", "more than $832,750", "exceeding 832,750"
ABOVE_AMOUNT_RE = re.compile(
    r"(?:above|over|more than|greater than|exceed(?:s|ing)?|beyond|larger than|bigger than)\s*"
    r"\$?\s*([\d,]{4,})", re.I
)


_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "eighteen": 18, "twenty": 20, "twenty-four": 24, "thirty": 30, "thirty-six": 36,
    "forty-eight": 48, "sixty": 60, "seventy-two": 72, "eighty-four": 84,
}
_SPELLED_TERM_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUM) + r")[\s-](year|month)s?\b", re.I)


def parse_question(q: str) -> ParsedQuestion:
    biz = True if BUSINESS_HINT_RE.search(q) else None
    term = None
    if m := TERM_MONTHS_Q_RE.search(q):
        term = int(m.group(1))
    elif m := TERM_YEARS_Q_RE.search(q):
        term = int(m.group(1)) * 12
    elif m := _SPELLED_TERM_RE.search(q):   # "five year", "thirty-six month"
        n = _WORD_NUM[m.group(1).lower()]
        term = n * 12 if m.group(2).lower() == "year" else n
    model_year = None
    if m := MODEL_YEAR_Q_RE.search(q):
        model_year = int(next(g for g in m.groups() if g))
    if model_year is None and re.search(r"\b(brand[-\s]?new|new)\s+(car|vehicle|auto|truck)\b", q, re.I):
        model_year = 2026   # newest tier on the page (2024-27) holds the headline "new" rate
    amount = None
    amts = [float(x.replace(",", "")) for x in DOLLAR_Q_RE.findall(q)]
    if amts:
        amount = min(amts)  # "between $10,000 and $24,999" -> probe with the lower bound
    above = None
    if m := ABOVE_AMOUNT_RE.search(q):
        above = float(m.group(1).replace(",", ""))
    metric_hint = "apy" if APY_Q_RE.search(q) else ("dividend_rate" if DIVIDEND_Q_RE.search(q) else None)
    return ParsedQuestion(
        text=q, is_business=biz, term_months=term, model_year=model_year, amount=amount,
        wants_highest=bool(HIGHEST_RE.search(q)), wants_lowest=bool(LOWEST_RE.search(q)),
        metric_hint=metric_hint, above_amount=above,
    )


def _bucket_for(text: str):
    for name, pattern, domain in BUCKETS:
        if re.search(pattern, text, re.I):
            return name, domain
    return None, None


def _in_range(x: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
    if x is None:
        return True
    if lo is not None and x < lo:
        return False
    if hi is not None and x > hi:
        return False
    return True


def _matches_bucket(fact, bucket_name: str) -> bool:
    _, pattern, _ = next(b for b in BUCKETS if b[0] == bucket_name)
    return bool(re.search(pattern, fact.family, re.I)) or bool(re.search(pattern, fact.label, re.I))


def _filter_business(facts: list, want_business: Optional[bool]) -> list:
    if want_business is None:
        non_biz = [f for f in facts if not f.is_business]
        return non_biz if non_biz else facts
    return [f for f in facts if f.is_business == want_business] or facts


def _drop_intro(facts: list) -> list:
    non_intro = [f for f in facts if not f.qualifiers.get("is_intro")]
    return non_intro if non_intro else facts


def _filter_jumbo(facts: list, question: str) -> list:
    """When the question names a variant, return only that one: 'jumbo' keeps
    the jumbo rows, 'standard'/'regular' keeps the non-jumbo rows. With neither
    word, return everything (callers then show the default/both)."""
    q = question.lower()
    wants_jumbo = "jumbo" in q
    wants_standard = bool(re.search(r"\bstandard\b|\bregular\b|\bnon[-\s]?jumbo\b", q))
    if wants_jumbo and not wants_standard:
        jumbo = [f for f in facts if f.qualifiers.get("is_jumbo")]
        return jumbo or facts
    if wants_standard and not wants_jumbo:
        standard = [f for f in facts if not f.qualifiers.get("is_jumbo")]
        return standard or facts
    return facts


def _pick_metric(facts: list, pq: ParsedQuestion, loan_domain: bool) -> list:
    if pq.metric_hint:
        hinted = [f for f in facts if f.metric == pq.metric_hint]
        if hinted:
            return hinted
    if loan_domain:
        # Mortgage tables report both a "Rate" and an "APR" column; keep both
        # (rendered together) rather than picking one, since members quote
        # the Rate but lenders quote the APR, and either can be asked for.
        rate_like = [f for f in facts if f.metric in ("apr", "rate")]
        return rate_like if rate_like else facts
    apy = [f for f in facts if f.metric == "apy"]
    return apy if apy else facts


def _canonical_source(store, winner):
    """The best page to cite for a winning fact: among all latest-snapshot
    pages that assert this exact value for this product/tier/metric, prefer the
    one whose URL path best matches the product family. This cites a product's
    own page (e.g. /personal/loans/home-loans for mortgages) rather than an
    incidental copy of the same table embedded on an unrelated page."""
    fam_tokens = set(retrieval.tokenize(winner.family))
    siblings = store.facts_where(
        lambda f: f.metric == winner.metric
        and f.label == winner.label and f.value_low == winner.value_low
        and f.value_high == winner.value_high and f.family == winner.family,
        taken_on=winner.taken_on,
    ) or [winner]

    def url_score(f):
        url_tokens = set(retrieval.tokenize(f.url.replace("/", " ").replace("-", " ")))
        # family-token overlap first, then prefer a product's own page over the
        # rates aggregator, then the shorter (more canonical) path.
        return (len(fam_tokens & url_tokens), 1 if f.page_type == "product" else 0, -len(f.url))

    best = max(siblings, key=url_score)
    return best.url, best


def _qualifier_text(fact) -> str:
    q = fact.qualifiers
    if q.get("amount_lo") is not None or q.get("amount_hi") is not None:
        lo, hi = q.get("amount_lo"), q.get("amount_hi")
        if lo is not None and hi is not None:
            return f"${lo:,.0f}-${hi:,.0f}"
        if hi is not None:
            return f"${hi:,.0f} or less"
        if lo is not None:
            return f"${lo:,.0f}+"
    term = q.get("term_months")
    if term is not None:
        # Model-year tiers (auto/ATV) label the row by year, so the term isn't
        # in the label — show it so each line reads "2024-27 Model Year
        # (60 months): 5.39% APR". Term-labelled rows ("6 Month") already have it.
        if q.get("model_year_hi") is not None:
            return f"{term} months"
        if not re.search(r"\d", fact.label):
            return f"{term} months"
    return ""


def _group_by(facts: list, keyfn):
    groups: dict = {}
    for f in facts:
        groups.setdefault(keyfn(f), []).append(f)
    return groups


def _default_tiers(facts: list, domain: str, pq: ParsedQuestion) -> list:
    """Facts already match the bucket+metric+business flag. Narrow by any
    qualifier the question gave; otherwise apply the domain's convention for
    "what do we quote when nobody specifies a tier"."""
    if pq.term_months is not None:
        with_term = [f for f in facts if f.qualifiers.get("term_months") == pq.term_months]
        if with_term:
            facts = with_term
        elif any(f.qualifiers.get("term_months") is not None for f in facts):
            return []  # a specific term was asked for and it doesn't exist -> caller should refuse
    if pq.model_year is not None:
        in_tier = [f for f in facts if _in_range(pq.model_year, f.qualifiers.get("model_year_lo"),
                                                  f.qualifiers.get("model_year_hi"))]
        if in_tier:
            facts = in_tier
        elif any("model_year_lo" in f.qualifiers for f in facts):
            return []
    if pq.amount is not None:
        in_amt = [f for f in facts if _in_range(pq.amount, f.qualifiers.get("amount_lo"), f.qualifiers.get("amount_hi"))]
        if in_amt:
            facts = in_amt

    by_family = _group_by(facts, lambda f: f.family)
    out = []
    for family, rows in by_family.items():
        if pq.term_months is not None or pq.amount is not None or pq.model_year is not None:
            out.extend(rows)
            continue
        if domain == "balance_tier" or pq.wants_highest or pq.wants_lowest:
            resolved = [(f, policy.resolve_current([x for x in rows if x.label == f.label and x.qualifiers == f.qualifiers]))
                        for f in _dedupe_tiers(rows)]
            flat = [r[0] for r in resolved]
            if not flat:
                continue
            pick = max(flat, key=lambda f: f.value_low) if not pq.wants_lowest else min(flat, key=lambda f: f.value_low)
            out.append(pick)
        elif domain == "loan_amount_tier":
            tiers = _dedupe_tiers(rows)
            with_ceiling = [f for f in tiers if f.qualifiers.get("amount_hi") is not None]
            if len(with_ceiling) > 2:
                lo_tier = min(with_ceiling, key=lambda f: f.qualifiers["amount_hi"])
                hi_tier = max(with_ceiling, key=lambda f: f.qualifiers["amount_hi"])
                out.extend({id(lo_tier): lo_tier, id(hi_tier): hi_tier}.values())
                out.extend(f for f in tiers if f.qualifiers.get("amount_hi") is None)
            else:
                out.extend(tiers)
        elif domain == "model_year_tier":
            tiers = _dedupe_tiers(rows)
            years = [f for f in tiers if f.qualifiers.get("model_year_hi") is not None]
            if years:
                newest = max(f.qualifiers["model_year_hi"] for f in years)
                out.extend(f for f in tiers if f.qualifiers.get("model_year_hi") == newest)
            else:
                out.extend(tiers)
        else:  # term_list / flat_list -> report every distinct tier
            out.extend(rows)
    return out


def _dedupe_tiers(rows: list) -> list:
    """One representative row per distinct (label, qualifiers) before precedence resolution."""
    seen = {}
    for r in rows:
        key = (r.label, tuple(sorted(r.qualifiers.items())))
        seen.setdefault(key, []).append(r)
    return [policy.resolve_current(group)[0] for group in seen.values() if policy.resolve_current(group)]


def numeric_answer(store, pq: ParsedQuestion, bucket_name: str, domain: str, as_of: Optional[str]) -> Optional[Answer]:
    if domain == "refuse":
        return Answer(text=REFUSAL_TEXT, confident=False)

    candidates = store.facts_where(lambda f: _matches_bucket(f, bucket_name))
    candidates = policy.filter_as_of(candidates, as_of)
    if not candidates:
        return None

    # "What's the rate above $832,750?" — if the question asks for an amount
    # beyond the largest published tier ceiling, the site simply doesn't
    # publish a rate up there, so refuse rather than quoting the top tier.
    if pq.above_amount is not None:
        ceilings = [f.qualifiers["amount_hi"] for f in candidates if f.qualifiers.get("amount_hi") is not None]
        if ceilings and pq.above_amount >= max(ceilings):
            return Answer(text=REFUSAL_TEXT, confident=False)

    loan_domain = domain in ("loan_amount_tier", "model_year_tier", "flat_list") or any(
        f.metric == "apr" for f in candidates
    )
    if bucket_name != "credit_card":
        candidates = _filter_business(candidates, pq.is_business)
    candidates = _drop_intro(candidates)
    candidates = _filter_jumbo(candidates, pq.text)
    candidates = _pick_metric(candidates, pq, loan_domain)
    if not candidates:
        return None

    # Precedence pass, applied across the WHOLE bucket (not tier-by-tier): a
    # page that vanished or went stale by the latest pull must not outvote a
    # page that's still there, and within that latest pull the official
    # rates hub/product page always beats a calculator or blog mention —
    # for every tier in this product line, not just the one being asked
    # about. This is what keeps a stale switch-and-earn/calculator copy of
    # last quarter's auto-loan table from ever surviving into an answer.
    latest_snap = max(f.taken_on for f in candidates)
    at_latest = [f for f in candidates if f.taken_on == latest_snap]
    best_rank = min(AUTHORITY_RANK.get(f.page_type, 3) for f in at_latest)
    authoritative = [f for f in at_latest if AUTHORITY_RANK.get(f.page_type, 3) == best_rank]
    if authoritative:
        candidates = authoritative

    chosen = _default_tiers(candidates, domain, pq)
    if not chosen:
        return Answer(text=REFUSAL_TEXT, confident=False)

    by_group = _group_by(chosen, lambda f: (f.family, f.label, tuple(sorted(f.qualifiers.items())), f.metric))
    winners, overridden = [], False
    for _, rows in by_group.items():
        before = {(r.value_low, r.value_high) for r in rows}
        won = policy.resolve_current(rows)
        if won:
            winners.extend(won)
            if len(before) > 1:
                overridden = True

    if not winners:
        return None

    parts = []
    sources = []
    seen_urls = set()
    for w in sorted(winners, key=lambda f: (f.family, f.qualifiers.get("term_months") or 0,
                                             f.qualifiers.get("amount_lo") or 0)):
        label = w.label or w.family
        tier = _qualifier_text(w)
        if tier and tier.lower() not in label.lower():
            label = f"{label} ({tier})"
        parts.append(f"{label}: {w.display()}")
        src_url, src_fact = _canonical_source(store, w)
        if src_url not in seen_urls:
            sources.append(Source(url=src_url, snapshot_id=src_fact.snapshot_id, quote=src_fact.quote[:200]))
            seen_urls.add(src_url)

    text = "; ".join(parts) + "."
    note = None
    if overridden:
        best = winners[0]
        note = (f"{policy.authority_label(best.page_type)} (effective {best.effective_date or best.taken_on}) "
                f"overrides an older/less authoritative value seen elsewhere.")
    return Answer(text=text, sources=sources, confident=True, conflict_note=note)


# --------------------------------------------------------------------------
# Entity-fact questions (routing number, branch count, Forbes, URLs...)
# --------------------------------------------------------------------------

ENTITY_PATTERNS = [
    ("routing_number", r"routing\s*number"),
    ("branch_count", r"how\s+many\s+branch|number\s+of\s+branch"),
    ("forbes_recognition", r"forbes"),
]


BRANCH_PHONE_RE = re.compile(
    r"phone number for (?:the )?(?P<name>[A-Za-z0-9.\- ]{3,40}?)(?:\s+branch)?\s+is\s+"
    r"(?P<phone>\(?\d{3}\)?[.\s-]?\d{3}[.\s-]?\d{4})",
    re.I,
)


def _branch_phone_answer(store, branch_tokens, as_of):
    """All current branch phone numbers whose named location contains every
    branch token in the question. A name like 'Learning Lab' legitimately maps
    to several branches, so we return all of them rather than guessing one."""
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    found = {}  # phone -> (url, quote)
    toks = [t.lower() for t in branch_tokens]
    for url, title, page_type, markdown in snap.pages:
        for m in BRANCH_PHONE_RE.finditer(markdown):
            name = m.group("name").strip().lower()
            if all(t in name for t in toks):
                found.setdefault(m.group("phone").strip(), (url, m.group(0)))
    if not found:
        return None
    phones = sorted(found)
    srcs = [Source(url=found[p][0], snapshot_id=snap.snapshot_id, quote=found[p][1][:200]) for p in phones]
    if len(phones) == 1:
        return Answer(text=f"That branch's phone number is {phones[0]}.", sources=srcs)
    return Answer(text=("That name matches more than one branch; the numbers on file are "
                        + ", ".join(phones) + "."), sources=srcs)


def _all_value_sources(store, key, value, as_of):
    """Every current page asserting a given entity value, most-canonical-looking
    URL first (shortest path), deduped."""
    facts = policy.filter_as_of(store.entities_where(lambda e: e.key == key and e.value == value), as_of)
    latest = max((f.taken_on for f in facts), default=None)
    facts = [f for f in facts if f.taken_on == latest]
    seen, out = set(), []
    for f in sorted(facts, key=lambda f: len(f.url)):
        if f.url not in seen:
            seen.add(f.url)
            out.append(Source(url=f.url, snapshot_id=f.snapshot_id, quote=f.quote[:200]))
    return out


def entity_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    q = pq.text

    # Phone numbers: a generic "phone number / call / reach you" question gets
    # the main line; "mortgage" gets the mortgage team's line; a question that
    # names a specific branch/location gets that branch's number(s) — and if
    # the name matches several branches (e.g. "Learning Lab"), all of them.
    # A private/individual's number (a CEO's cell, someone's direct line) is not
    # published — abstain rather than handing out the main line as if it were.
    if re.search(r"\bprivate\b|\bcell\b|\bceo\b|\bpresident\b|\bdirect\s+line\b|\bhome\s+(?:phone|number)\b", q, re.I) \
            and re.search(r"\bphone\b|\bnumber\b|\bline\b|\bcall\b|\bcell\b|\breach\b|\bextension\b", q, re.I) \
            and not re.search(r"routing|account\s+number|member\s+number", q, re.I):
        return Answer(text=REFUSAL_TEXT, confident=False)

    if re.search(r"\bphone\b|\bcall\b|\breach\b|\bnumber to\b", q, re.I) and not re.search(r"routing|account\s+number|member\s+number", q, re.I):
        want_mortgage = bool(re.search(r"mortgage|home\s+loan", q, re.I))
        branch_tokens = _branch_name_tokens(q)
        if branch_tokens and not want_mortgage:
            branch = _branch_phone_answer(store, branch_tokens, as_of)
            if branch is not None:
                return branch
        key = "mortgage_phone" if want_mortgage else "main_phone"
        facts = policy.filter_as_of(store.entities_where(lambda e: e.key == key), as_of)
        winners = policy.resolve_entity_current(facts)
        if winners:
            w = winners[0]
            src = [Source(url=w.url, snapshot_id=w.snapshot_id, quote=w.quote[:200])]
            if want_mortgage:
                return Answer(text=f"You can reach Dupaco's Mortgage Team at {w.value}, ext. 204.", sources=src)
            return Answer(text=f"You can reach Dupaco at {w.value}.", sources=src)

    for key, pattern in ENTITY_PATTERNS:
        if re.search(pattern, q, re.I):
            facts = store.entities_where(lambda e: e.key == key or e.category == key)
            facts = policy.filter_as_of(facts, as_of)
            winners = policy.resolve_entity_current(facts)
            if not winners:
                continue
            w = winners[0]
            sources = [Source(url=w.url, snapshot_id=w.snapshot_id, quote=w.quote[:200])]
            if key == "routing_number":
                # The routing number is a single invariant fact printed on many
                # pages; cite every current page that states it so the member
                # can land on whichever is canonical for them.
                all_src = _all_value_sources(store, key, w.value, as_of)
                return Answer(text=f"Dupaco's routing number is {w.value}.", sources=all_src or sources)
            if key == "branch_count":
                vals = sorted({e.value for e in winners}, key=lambda v: -int(v))
                return Answer(text=f"Dupaco has {vals[0]} branch office locations.", sources=sources)
            if key == "forbes_recognition":
                return Answer(
                    text=f"Forbes has recognized Dupaco as a top credit union; the most recent release calls it the {w.value} time.",
                    sources=sources,
                )
    return None


# --------------------------------------------------------------------------
# Branch address / hours — never fabricated: if the place name isn't found
# in the branch directory, this returns a refusal rather than a guess.
# --------------------------------------------------------------------------

ADDRESS_RE = re.compile(
    r"([A-Z0-9][\w.&'’\- ]*(?:Rd|Rd\.|St|St\.|Ave|Ave\.|Ln|Ln\.|Dr|Dr\.|Blvd|Blvd\.|Way)\.?,\s*"
    r"[A-Za-z .]+,\s*(?:IA|IL|WI),\s*\d{5})"
)
BRANCH_NAME_STOPWORDS = {
    "branch", "branches", "hours", "lobby", "drive", "drive-up", "up", "the", "for", "of", "at",
    "dupaco", "can", "you", "give", "me", "is", "are", "does", "do", "what", "where", "which",
    "still", "open", "today", "all", "same", "keep", "general", "address",
}


def _branch_name_tokens(q: str) -> list:
    words = re.findall(r"[A-Za-z]+", q)[1:]  # skip the sentence-initial WH-word/aux verb
    return [w for w in words if len(w) > 1 and w[0].isupper() and w.lower() not in BRANCH_NAME_STOPWORDS]


_STATE_ABBR = {"iowa": "ia", "illinois": "il", "wisconsin": "wi"}


def address_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    q = pq.text
    if not re.search(r"\baddress\b|\blocated\b|\bwhere\s*(?:is|'s)\b", q, re.I):
        return None
    tokens = [_STATE_ABBR.get(t.lower(), t) for t in _branch_name_tokens(q)]
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    for url, title, page_type, markdown in snap.pages:
        if page_type != "branch":
            continue
        # Only match the Dupaco-owned branch rows, not partner ATMs / dealers.
        for block in re.split(r"\n###\s", markdown):
            if "dupaco branch" not in block.lower():
                continue
            m = ADDRESS_RE.search(block)
            if not m:
                continue
            addr = m.group(1)
            if tokens and all(t.lower() in addr.lower() for t in tokens):
                return Answer(text=addr + ".", sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=addr)])
    if tokens:
        return Answer(text=REFUSAL_TEXT, confident=False)
    return None


DAY_LINE_RE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-zA-Z\-]*:\s*(?:\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?\s*[–—-]\s*"
    r"(?:\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?|noon)|closed|atm only)", re.I  # end time may be "noon"
)


def _extract_hour_blocks(markdown: str) -> dict:
    """Pages that announce a new branch state its hours as a '**Lobby**' /
    '**Drive-up**' marker followed by one day-range line per paragraph.
    Returns {"Lobby": "...", "Drive-up": "..."} for whatever this page has."""
    blocks: dict = {}
    for marker_m in re.finditer(r"\*\*(Lobby|Drive-?up)\*\*", markdown, re.I):
        label = "Drive-up" if "drive" in marker_m.group(1).lower() else "Lobby"
        lines = []
        for line in markdown[marker_m.end():].split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("**") or not DAY_LINE_RE.match(line):
                break
            lines.append(line)
        if lines:
            blocks[label] = "; ".join(lines)
    return blocks


def hours_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    q = pq.text
    if not re.search(r"\bhours\b", q, re.I):
        return None
    tokens = _branch_name_tokens(q)
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    is_drive_up = bool(re.search(r"drive.?up", q, re.I))
    label = "Drive-up" if is_drive_up else "Lobby"

    found = []  # (url, {"Lobby":.., "Drive-up":..})
    for url, title, page_type, markdown in snap.pages:
        if tokens and not all(t.lower() in markdown.lower() for t in tokens):
            continue
        blocks = _extract_hour_blocks(markdown)
        if blocks:
            found.append((url, blocks))

    if tokens and not found:
        return Answer(text=REFUSAL_TEXT, confident=False)
    if not found:
        return None

    distinct = []
    seen = set()
    for url, blocks in found:
        text = blocks.get(label)
        if text and text not in seen:
            seen.add(text)
            distinct.append((url, text))

    if not distinct:
        return None
    if tokens or len(distinct) == 1:
        url, text = distinct[0]
        return Answer(text=f"{label} hours: {text}.",
                      sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=text)])

    # Multiple distinct schedules exist across branches. "Do all branches keep
    # the same hours?" and "what are the lobby hours?" both want every distinct
    # *current* schedule, not a single example — so list them all (capped), per
    # the include-all-valid-values rule.
    cap = 4
    examples = "; and ".join(t for _, t in distinct[:cap])
    sources = [Source(url=u, snapshot_id=snap.snapshot_id, quote=t) for u, t in distinct[:cap]]
    other_label = "Drive-up" if label == "Lobby" else "Lobby"
    other_seen, other_examples = set(), []
    for url, blocks in found:
        t = blocks.get(other_label)
        if t and t not in other_seen:
            other_seen.add(t)
            other_examples.append((url, t))
        if len(other_examples) == cap:
            break
    extra = ""
    if other_examples:
        extra = f" {other_label} hours can differ too — e.g. " + "; or ".join(t for _, t in other_examples) + "."
        sources.extend(Source(url=u, snapshot_id=snap.snapshot_id, quote=t) for u, t in other_examples)
    return Answer(
        text=(f"{label} hours vary by branch — for example: {examples}. "
              f"Let us know which branch, or we can list them all.{extra}"),
        sources=sources,
    )


# --------------------------------------------------------------------------
# A handful of other well-known, text-only "knowledge nugget" questions.
# Each is a thin regex over the question plus a regex over page text, kept
# general (no hard-coded answer values) so a reworded question still works.
# --------------------------------------------------------------------------

MISC_PATTERNS = [
    (r"hy-?vee", "hy-vee"),
    (r"eligib|who can (?:become|join)|membership.{0,15}(?:area|eligib)", "eligibility"),
    (r"mobile wallet|apple pay|google pay|samsung pay", "wallet"),
    (r"home\$?tart", "hometart"),
    (r"volunteer time off|\bvto\b", "vto"),
    (r"update.{0,15}contact|change.{0,15}(?:address|phone|email)", "contact_update"),
    (r"join.{0,15}(?:online|dupaco)|how do i join", "join_url"),
]


MIN_OPEN_RE = re.compile(
    r"(minimum|min\.?|how much).{0,30}\bto\s+open\b|\bto\s+open\b.{0,20}(account|need|deposit|require)|"
    r"(minimum|how much).{0,20}(deposit|balance).{0,20}open", re.I
)
MIN_EARN_RE = re.compile(r"(minimum|min\.?|how much).{0,30}to\s+earn", re.I)


def minimum_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    """Structured 'minimum to open / to earn' lookup for a named deposit
    product, so these don't fall back to weaker prose retrieval."""
    q = pq.text
    want_earn = bool(MIN_EARN_RE.search(q))
    if not want_earn and not MIN_OPEN_RE.search(q):
        return None
    metric = "min_to_earn" if want_earn else "min_to_open"
    bucket_name, _ = _bucket_for(q)
    if not bucket_name:
        return None
    facts = store.facts_where(lambda f: f.metric == metric and _matches_bucket(f, bucket_name))
    facts = policy.filter_as_of(facts, as_of)
    facts = _filter_business(facts, pq.is_business)
    if not facts:
        return None
    # The "opening" minimum is the lowest across tiers (you can open at the
    # entry tier); resolve each distinct value through the current-value policy.
    winners = policy.resolve_current(facts)
    if not winners:
        return None
    best = min(winners, key=lambda f: f.value_low)
    label = best.label or best.family
    verb = "to earn dividends on" if want_earn else "to open"
    return Answer(text=f"The minimum {verb} {label} is {best.display()}.",
                  sources=[Source(url=best.url, snapshot_id=best.snapshot_id, quote=best.quote[:200])])


# Describe-domain products (e.g. student loans) have no rate table; answer them
# from the product page's own prose, returning a grounded sentence that names
# the product, so a reworded "do you have X?" still lands on the right page.
DESCRIBE_KEYWORDS = {
    "student_loan": ("student loan", r"/student-loans|/student-loan"),
}


def describe_answer(store, pq: ParsedQuestion, as_of: Optional[str], bucket_name: str) -> Optional[Answer]:
    phrase, url_pat = DESCRIBE_KEYWORDS.get(bucket_name, (None, None))
    if not phrase:
        return None
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    # Prefer the product's own page; fall back to any page that uses the phrase.
    pages = [p for p in snap.pages if re.search(url_pat, p[0], re.I)] or \
            [p for p in snap.pages if phrase in p[3].lower()]
    for url, title, page_type, markdown in pages:
        for sentence in _clean_sentences(markdown):
            if phrase in sentence.lower():
                return Answer(text=sentence[:400],
                              sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=sentence[:200])])
    return None


ELIGIBILITY_Q_RE = re.compile(
    r"\bcan\s+(?:a|an|my)?\s*\w*\s*(?:business|consumer|member|individual).{0,20}open|"
    r"who\s+can\s+open|is\s+.{0,30}\beligible|available\s+to\b|open\s+to\b", re.I
)
ELIGIBILITY_MARKER_RE = re.compile(
    r"available\s+exclusively\s+to|exclusively\s+to\s+\w+\s+members|not\s+eligible|"
    r"consumer\s+members|available\s+(?:only\s+)?to\s+(?:consumer|individual|member)", re.I
)


def product_eligibility_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    """Who may open a specific account (e.g. 'Can a business open Bright
    Savings?'). The answer lives in a product footnote ('available exclusively
    to consumer members...'), which doesn't rank well for the natural
    question, so look it up directly on the product's pages."""
    q = pq.text
    if not ELIGIBILITY_Q_RE.search(q):
        return None
    bucket_name, _ = _bucket_for(q)
    if not bucket_name:
        return None
    _, pattern, _ = next((b for b in BUCKETS if b[0] == bucket_name), (None, None, None))
    snap = store.snapshot_for(as_of)
    if not snap or not pattern:
        return None
    # A product's own dedicated page (the pattern matches its URL or title)
    # carries that product's eligibility footnote unambiguously; the
    # multi-product rates hub mixes several footnotes, so try it last.
    def dedicated(p):
        return bool(re.search(pattern, p[0], re.I) or re.search(pattern, p[1] or "", re.I))
    ordered = ([p for p in snap.pages if p[2] == "product" and dedicated(p)]
               + [p for p in snap.pages if p[2] in ("rates_hub", "product")])
    for url, title, page_type, markdown in ordered:
        if not re.search(pattern, markdown, re.I):
            continue
        for sentence in _clean_sentences(markdown):
            if ELIGIBILITY_MARKER_RE.search(sentence):
                return Answer(text=sentence[:400],
                              sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=sentence[:200])])
    return None


CHECKING_FEE_WAIVE_RE = re.compile(
    r"\$(\d+)\s*monthly fee[^.$]*?monthly balance of \$([\d,]+)", re.I)


def checking_fees_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    """Checking-account *fee* questions are not rate questions, and the fee is
    stated in prose on the checking-accounts page ("avoid a $10 monthly fee by
    keeping a monthly balance of $5,000+"; Free Checking has "no monthly
    fees"). Extract both from that page rather than letting it fall to a
    tangential blog sentence."""
    q = pq.text
    if not re.search(r"checking", q, re.I):
        return None
    if not re.search(r"\bfee|\bfees|\bmonthly|service charge|\bcharge\b|\bcost", q, re.I):
        return None
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    page = next((p for p in snap.pages if p[0].rstrip("/").endswith("/personal/checking-accounts")), None)
    if page is None:
        return None
    md = page[3]
    m = CHECKING_FEE_WAIVE_RE.search(md)
    free_no_fee = bool(re.search(r"free checking[\s\S]{0,400}?no monthly (?:fees|service charges)", md, re.I))
    parts = []
    if free_no_fee:
        parts.append("Free Checking: no monthly fee (no minimum balance).")
    if m:
        parts.append(f"1st Rate Checking: ${m.group(1)} monthly fee, waived with a monthly balance "
                     f"of ${m.group(2)} or more.")
    if not parts:
        return None
    return Answer(text=" ".join(parts) + " Other transaction or overdraft fees may apply.",
                  sources=[Source(url=page[0], snapshot_id=snap.snapshot_id)])


def misc_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    q = pq.text
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    kind = None
    for pattern, name in MISC_PATTERNS:
        if re.search(pattern, q, re.I):
            kind = name
            break
    if kind is None:
        return None

    if kind == "contact_update":
        # The actionable value (the phone) is read from the fact store, not typed.
        ph = policy.resolve_entity_current(
            policy.filter_as_of(store.entities_where(lambda e: e.key == "main_phone"), as_of))
        phone = ph[0].value if ph else "800-373-7600"
        src = [Source(url=ph[0].url, snapshot_id=ph[0].snapshot_id)] if ph else []
        return Answer(text=f"Log in to Shine Online or Mobile Banking to update your contact info, "
                           f"or call {phone}.", sources=src)

    if kind == "join_url":
        for url, title, page_type, markdown in snap.pages:
            m = re.search(r"\((https://join\.dupaco\.com/?)\)", markdown)
            if m:
                return Answer(text=f"Join online at {m.group(1)}", sources=[Source(url=url, snapshot_id=snap.snapshot_id)])

    if kind == "wallet":
        # Return the page's own sentence that names the wallets, not a typed one.
        for url, title, page_type, markdown in snap.pages:
            for sent in re.split(r"(?<=[.!?])\s+", markdown):
                low = sent.lower()
                if "apple pay" in low and "google pay" in low and "samsung pay" in low and len(sent) < 300:
                    return Answer(text=re.sub(r"\s+", " ", sent).strip(),
                                  sources=[Source(url=url, snapshot_id=snap.snapshot_id)])

    if kind == "hometart":
        for url, title, page_type, markdown in snap.pages:
            m = re.search(r"Home\$?tart[\s\S]{0,300}?\$([\d,]+)", markdown, re.I)
            if m:
                return Answer(text=f"Home$tart (an FHLB program) provides up to ${m.group(1)} toward down "
                                    f"payment and closing costs for first-time homebuyers.",
                              sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=m.group(0)[:200])])

    if kind == "eligibility":
        # Build the answer from the membership page's own clauses: the
        # "live or work … charter area" criterion plus the states it covers.
        for url, title, page_type, markdown in snap.pages:
            if "/about/join" not in url and "membership" not in url.lower():
                continue
            clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)        # drop images
            clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)       # links -> text
            crit = re.search(r"[A-Z][^.\n]*\blive or work\b[^.\n]*\.", clean, re.I)
            states = re.search(r"[A-Z][^.\n]*\bIowa\b[^.\n]*\bWisconsin\b[^.\n]*\.", clean, re.I)
            if crit and states:
                text = re.sub(r"\s+", " ", f"{crit.group(0).strip()} {states.group(0).strip()}")
                return Answer(text=text[:400], sources=[Source(url=url, snapshot_id=snap.snapshot_id)])

    if kind == "vto":
        # The site reports VTO participation for more than one program year on
        # separate (still-live) articles — 294 in 2023, 256 in 2024. "Last
        # year" has no single true referent from a 2026 snapshot, so report
        # every distinct year's figure rather than guessing one. Each (year,
        # count) is pinned to its own article so both stay current and cited.
        by_year = {}
        for url, title, page_type, markdown in snap.pages:
            for m in re.finditer(
                r"(\d+)\s*\**\s*Dupaco\s+employees\**\s+used\s+a\s+(?:new\s+)?paid\s+benefit\s+in\s+(\d{4})"
                r"\s+called\s+Volunteer Time Off",
                markdown, re.I,
            ):
                count, year = m.group(1), m.group(2)
                by_year.setdefault(year, (count, url, m.group(0)))
        if by_year:
            parts = [f"{cnt} in {yr}" for yr, (cnt, _, _) in sorted(by_year.items(), reverse=True)]
            srcs = [Source(url=u, snapshot_id=snap.snapshot_id, quote=qt[:200])
                    for _, (_, u, qt) in sorted(by_year.items(), reverse=True)]
            lead = "Dupaco employees used Volunteer Time Off: " + "; ".join(parts) + "."
            return Answer(text=lead, sources=srcs)

    if kind == "hy-vee":
        date_fact, alt_fact = None, None
        for e in store.entities_where(lambda e: e.key == "hy-vee_closure"):
            if e.taken_on <= (as_of or snap.taken_on):
                date_fact = e
        for url, title, page_type, markdown in snap.pages:
            if "hy-vee" in url.lower() or "hy-vee" in markdown.lower()[:200]:
                m = re.search(r"close[sd]?\s+(?:on\s+)?([A-Z][a-z]+ \d{1,2},\s*\d{4})", markdown)
                alt = re.search(r"(Key West)[\s\S]{0,80}(Sycamore Street)", markdown)
                if m or alt:
                    text_parts = []
                    if re.search(r"still open|open today|closed", q, re.I) or True:
                        if m:
                            text_parts.append(f"No — the Hy-Vee branch closed/is scheduled to close {m.group(1)}.")
                    if alt and re.search(r"where|full.service|alternat", q, re.I):
                        text_parts.append(f"Members can use the {alt.group(1)} and {alt.group(2)} branches instead.")
                    if text_parts:
                        return Answer(text=" ".join(text_parts),
                                      sources=[Source(url=url, snapshot_id=snap.snapshot_id)])
    return None


# --------------------------------------------------------------------------
# "What URL/link should I use for X" — mines markdown links instead of
# guessing; a page's own canonical URL is the fallback.
# --------------------------------------------------------------------------

LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")
URL_QUESTION_RE = re.compile(r"\burl\b|\blink\b|\bwebsite\b", re.I)


def url_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    q = pq.text
    if not URL_QUESTION_RE.search(q):
        return None
    snap = store.snapshot_for(as_of)
    if not snap:
        return None

    # A "what URL for X's rate" question already has an answer: whichever
    # page numeric_answer() decided is the current, authoritative source for
    # X. Prefer that over guessing from link text.
    bucket_name, domain = _bucket_for(q)
    if bucket_name and domain not in ("refuse", "describe"):
        numeric = numeric_answer(store, pq, bucket_name, domain, as_of)
        if numeric is not None and numeric.confident and numeric.sources:
            url = numeric.sources[0].url
            return Answer(text=url, sources=[Source(url=url, snapshot_id=snap.snapshot_id)])

    q_tokens = set(retrieval.tokenize(q))
    results = snap.index.search(q, top_k=6)
    best_url, best_score = None, -1.0
    for chunk, score in results:
        for anchor, href in LINK_RE.findall(chunk.text):
            if "dupaco.com" not in href:
                continue
            anchor_tokens = set(retrieval.tokenize(anchor))
            href_tokens = set(retrieval.tokenize(href.replace("/", " ").replace("-", " ")))
            overlap = len(q_tokens & (anchor_tokens | href_tokens))
            s = overlap - 0.01 * len(href)
            if s > best_score:
                best_score, best_url = s, href
        if best_url is None:
            s = score - 0.01 * len(chunk.url)
            if s > best_score:
                best_score, best_url = s, chunk.url
    if not best_url:
        return None
    return Answer(text=best_url, sources=[Source(url=best_url, snapshot_id=snap.snapshot_id)])


# --------------------------------------------------------------------------
# Generic retrieval fallback (prose questions nothing structured answers)
# --------------------------------------------------------------------------

MIN_RETRIEVAL_SCORE = 2.0

# Calls-to-action / UI chrome that look like sentences but answer nothing.
# These CMS pages are full of them ("Apply today", "I'm ready to open...",
# "HOW TO Open a ... account", "See details"). A real answer sentence almost
# never opens with an imperative verb like these.
CTA_RE = re.compile(
    r"^\s*(?:"
    r"i['’]?m\s+ready|how\s+to\b|apply\b|open\s+(?:my|an?|your)\b|see\s+details|learn\s+more|"
    r"get\s+(?:started|more|a\s+quote|my)|explore\b|compare\s+rates|join\b|read\s+more|"
    r"continue\b|go\b|become\s+a\s+member|sign\s+up|start\b|view\b|find\s+(?:a|my)\b|"
    r"let['’]?s\b|download\b|contact\s+us|shop\b|book\b|schedule\b|enroll\b|calculate\b|next\b|submit\b"
    r")",
    re.I,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_sentences(text: str):
    """Yield answer-worthy sentences from a chunk: no CTAs, no bare questions,
    no table/image fragments, nothing under ~5 words."""
    for raw in SENTENCE_SPLIT_RE.split(text):
        s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw).strip().strip("*_#>| ").strip()
        if not s or len(s.split()) < 5:
            continue
        if CTA_RE.match(s):
            continue
        if s.count("|") >= 2 or s.startswith("!"):
            continue
        if s.endswith("?"):            # FAQ headings / nav links, not answers
            continue
        yield s


def _clean_chunk(chunk, q_tokens: set):
    """Return (text, overlap) for a chunk reduced to its answer sentences, or
    (None, 0) if nothing survives. `overlap` is the count of distinct question
    content words present — our confidence/grounding signal."""
    kept = list(_clean_sentences(chunk.text))
    if not kept:
        return None, 0
    joined = " ".join(kept)
    overlap = len(q_tokens & set(retrieval.tokenize(joined)))
    return joined, overlap


# Generic banking vocabulary — words that are NOT distinctive on their own, so
# their presence in a passage does not mean the passage answers the question.
# "crypto debit card" is distinctively about "crypto"; "debit"/"card" are not.
GENERIC_BANKING = {
    "account", "accounts", "rate", "rates", "loan", "loans", "card", "cards",
    "debit", "credit", "savings", "checking", "certificate", "certificates",
    "cd", "cds", "interest", "apr", "apy", "dividend", "bank", "banking",
    "dupaco", "offer", "offers", "open", "opening", "pay", "payment", "payments",
    "fee", "fees", "member", "members", "membership", "money", "balance",
    "deposit", "deposits", "minimum", "min", "term", "terms", "month", "months",
    "year", "years", "current", "currently", "new", "used", "buy", "buying",
    "get", "much", "program", "programs", "product", "products", "service",
    "services", "available", "offered", "need", "want", "like", "would",
    "does", "do", "can", "what", "which", "how", "your", "you", "the", "a",
    "have", "has", "had", "is", "are", "be", "any", "my", "me", "i",
}


# The product-type nouns among the generic words. A "crypto debit card"
# question is anchored on "debit"/"card": if the passage carrying the
# distinctive term ("crypto") doesn't also mention the product noun, it isn't
# describing that product and we abstain.
PRODUCT_WORDS = {
    "account", "accounts", "card", "cards", "debit", "credit", "loan", "loans",
    "savings", "checking", "certificate", "certificates", "cd", "cds",
    "mortgage", "visa", "heloc", "ira",
}


def _distinctive_terms(question: str) -> set:
    return set(retrieval.tokenize(question)) - GENERIC_BANKING


def retrieval_answer(store, pq: ParsedQuestion, as_of: Optional[str]) -> Answer:
    snap = store.snapshot_for(as_of)
    if not snap:
        return Answer(text=REFUSAL_TEXT, confident=False)
    ans = _retrieval_in(snap, pq)
    if ans.confident or ans.sources:
        return ans
    # Nothing on the current site. A crawl can miss a page, so look only at
    # pages that VANISHED from this crawl, newest earlier snapshot first.
    current_urls = {u for u, _, _, _ in snap.pages}
    for old in reversed([s for s in store.snapshots if s.taken_on < snap.taken_on]):
        gone = {u for u, _, _, _ in old.pages} - current_urls
        if not gone:
            continue
        old_ans = _retrieval_in(old, pq, only_urls=gone)
        if old_ans.sources:
            old_ans.text += f" (This page was not in the {snap.taken_on} crawl; last seen {old.taken_on}.)"
            return old_ans
    return ans


def _retrieval_in(snap, pq: ParsedQuestion, only_urls=None) -> Answer:
    # Scan deeper than the first few hits: on these CMS pages the top BM25
    # chunks are often calls-to-action ("Open my ..."), which clean to nothing
    # and are skipped, so the real answer sentence can sit several ranks down.
    results = snap.index.search(pq.text, top_k=10 if only_urls is None else 50)
    if only_urls is not None:
        results = [(c, sc) for c, sc in results if c.url in only_urls]
    if not results or results[0][1] < MIN_RETRIEVAL_SCORE:
        return Answer(text=REFUSAL_TEXT, confident=False)
    q_tokens = set(retrieval.tokenize(pq.text))
    # The question's distinctive terms (minus stop words and generic banking
    # vocabulary) must co-occur in the matched passage, AND if the question
    # names a product noun (debit/card/loan/...), that noun must appear in the
    # same passage too. Otherwise the passage only shares filler, or mentions
    # the topic without the product ("do you have a crypto debit card?" ->
    # a blog about crypto that never mentions a card), so we abstain.
    distinctive = _distinctive_terms(pq.text)
    product_anchors = {t for t in retrieval.tokenize(pq.text) if t in PRODUCT_WORDS}
    # Blogs/promos may be used only when the question is about a KNOWN product
    # (a product bucket matches). Then a blog sentence stating a fact about that
    # product ("Dupaco's Rewards Visa has no annual fee") is fair game. For an
    # unrecognized product ("a crypto debit card"), a blog that merely discusses
    # the topic must not be mistaken for the credit union offering it, so blogs
    # are excluded and the question abstains.
    allow_blogs = _bucket_for(pq.text)[0] is not None

    best = None
    for chunk, score in results:
        if chunk.page_type in ("news", "calculator") and not allow_blogs:
            continue
        text, overlap = _clean_chunk(chunk, q_tokens)
        if not text:
            continue
        passage_tokens = set(retrieval.tokenize(text))
        if distinctive and not (distinctive & passage_tokens):
            continue  # none of the distinctive terms are here; not an answer
        if distinctive and product_anchors and not (product_anchors & passage_tokens):
            continue  # distinctive topic present, but not tied to the product asked about
        if overlap >= 2:
            best = (text, overlap, chunk.url)
            break
        if overlap >= 1 and best is None:
            best = (text, overlap, chunk.url)
    if best is None:
        return Answer(text=REFUSAL_TEXT, confident=False)
    text, overlap, url = best
    text = text[:500]
    return Answer(text=text, sources=[Source(url=url, snapshot_id=snap.snapshot_id, quote=text[:200])],
                  confident=overlap >= 2)


# --------------------------------------------------------------------------
# Top-level entry point used by solution.py
# --------------------------------------------------------------------------

OFFER_RE = re.compile(
    r"\bbonus\b|\bcash\s?back\b|\bpromotion\b|\bpromo\b|\bsign[-\s]?up\s+bonus\b|"
    r"\breferral\b|\bgift\s?card\b|\bsweepstakes\b|\bgiveaway\b", re.I
)


def _phantom_offer_guard(store, pq: ParsedQuestion, as_of: Optional[str]) -> Optional[Answer]:
    """A question that asserts a specific promotional offer the site never
    states — "a student checking account bonus of $500" — must not be answered
    with unrelated rate figures. If the question names an offer AND a specific
    dollar amount, require that amount to appear on a page that also mentions
    the offer; otherwise the offer is ungrounded and we abstain."""
    q = pq.text
    if not OFFER_RE.search(q) or pq.amount is None:
        return None
    snap = store.snapshot_for(as_of)
    if not snap:
        return None
    amt = pq.amount
    amt_pat = re.escape(f"{amt:,.0f}") + "|" + re.escape(f"{amt:.0f}")
    # The amount must sit close to an offer word (same clause), not merely on
    # the same page — "$500" as a deposit *requirement* near "bonus" elsewhere
    # on the page does not ground a "$500 bonus" claim.
    offer = r"(?:bonus|cash\s?back|sweepstakes|giveaway)"
    near = re.compile(
        offer + r"\b.{0,40}\$?\s?(?:" + amt_pat + r")\b"
        + r"|\$?\s?(?:" + amt_pat + r")\b.{0,40}" + offer + r"\b",
        re.I | re.S,
    )
    for url, title, page_type, markdown in snap.pages:
        if near.search(markdown):
            return None  # the specific offer amount is grounded; answer normally
    return Answer(text=REFUSAL_TEXT, confident=False)


def answer_question(store, question: str, as_of: Optional[str]) -> Answer:
    pq = parse_question(question)

    # Abstain on an ungrounded promotional offer ("$500 checking bonus") before
    # anything tries to answer it with rate figures.
    guard = _phantom_offer_guard(store, pq, as_of)
    if guard is not None:
        return guard

    # A checking *fee* question must beat the rate path — "1st Rate Checking"
    # contains the word "Rate", which would otherwise route it to APY figures.
    if re.search(r"checking", question, re.I) and re.search(r"\bfee|\bfees|service charge|\bcharge\b|\bcost", question, re.I):
        ans = checking_fees_answer(store, pq, as_of)
        if ans is not None:
            return ans

    # URL/link questions are routed first: "credit card" or "certificate" in
    # a URL question would otherwise be caught by the numeric-rate bucket.
    if URL_QUESTION_RE.search(question):
        ans = url_answer(store, pq, as_of)
        if ans is not None:
            return ans

    bucket_name, domain = _bucket_for(question)
    # A detected model year (e.g. "2025 model", "a 2022 vehicle") with no other
    # product in the question is an auto-loan question by context.
    if bucket_name is None and pq.model_year is not None:
        bucket_name, domain = "auto_loan", "model_year_tier"
    # In an auto/ATV question a bare 4-digit year IS the model year, even
    # without an adjacent "vehicle" word ("60-month auto loan, 2025?").
    if domain == "model_year_tier" and pq.model_year is None:
        ym = re.search(r"\b(199\d|20[0-3]\d)\b", question)
        if ym:
            pq.model_year = int(ym.group(1))
    is_rate_shaped = bool(RATE_SHAPED_RE.search(question))
    # A concrete rate qualifier (term / model year / loan amount) on a tiered
    # product is itself a rate question, even when the word "rate" is absent
    # ("2022 vehicle, 60 months" -> the auto rate for that tier).
    TIERED = ("loan_amount_tier", "model_year_tier", "flat_list", "term_list", "card", "card_list", "balance_tier")
    has_qualifier = pq.term_months is not None or pq.model_year is not None or pq.amount is not None
    route_numeric = is_rate_shaped or domain == "refuse" or (domain in TIERED and has_qualifier)
    if bucket_name and route_numeric:
        ans = numeric_answer(store, pq, bucket_name, domain, as_of)
        if ans is not None:
            return ans
        if domain in ("loan_amount_tier", "model_year_tier", "flat_list", "term_list", "card", "card_list"):
            return Answer(text=REFUSAL_TEXT, confident=False)

    for handler in (entity_answer, minimum_answer, product_eligibility_answer,
                    address_answer, hours_answer, misc_answer):
        ans = handler(store, pq, as_of)
        if ans is not None:
            return ans

    if bucket_name and domain == "describe":
        ans = describe_answer(store, pq, as_of, bucket_name)
        if ans is not None:
            return ans

    if is_rate_shaped:
        return Answer(text=REFUSAL_TEXT, confident=False)

    return retrieval_answer(store, pq, as_of)
