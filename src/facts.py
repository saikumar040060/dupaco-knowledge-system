"""
Turns raw Pages into two kinds of structured, versioned facts:

  FactRow    - a single numeric rate/APY/APR/fee value for one product+tier,
               taken from one markdown table row, on one page, in one snapshot.
  EntityFact - a non-numeric fact (routing number, phone, URL, branch count,
               address, hours, recognition count...) pulled from free text.

Nothing here decides what's "current" — that is policy.py's job. This module
only extracts and normalizes; it keeps every fact it finds, from every
snapshot, so history is never thrown away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from tables import extract_tables

# --------------------------------------------------------------------------
# Page classification
# --------------------------------------------------------------------------

PAGE_TYPE_RATES_HUB = "rates_hub"          # the /rates, /business-rates aggregators
PAGE_TYPE_PRODUCT = "product"              # /personal/..., /business/... product pages
PAGE_TYPE_CALCULATOR = "calculator"        # /calculators/..., *-calculator blog posts, switch-and-earn
PAGE_TYPE_NEWS = "news"                    # /newsroom/..., dated blog posts
PAGE_TYPE_BRANCH = "branch"                # /contact, locator
PAGE_TYPE_OTHER = "other"

# Authority order used by policy.py: lower number = more authoritative.
AUTHORITY_RANK = {
    PAGE_TYPE_RATES_HUB: 0,
    PAGE_TYPE_PRODUCT: 0,
    PAGE_TYPE_BRANCH: 1,
    PAGE_TYPE_CALCULATOR: 2,
    PAGE_TYPE_NEWS: 3,
    PAGE_TYPE_OTHER: 3,
}

_CALC_URL_HINTS = ("/calculators/", "calculator", "switch-and-earn", "/learn/blog/")
_NEWS_URL_HINTS = ("/newsroom/",)
_BRANCH_URL_HINTS = ("/contact",)
_RATES_HUB_HINTS = ("/rates", "/business-rates")


def classify_page(url: str, title: str) -> str:
    u = url.lower()
    t = (title or "").lower()
    if "promo" in u or "special-offer" in u or "promotion" in t:
        return PAGE_TYPE_CALCULATOR  # promo pages carry the same low authority as calculators
    if any(u.rstrip("/").endswith(h.rstrip("/")) for h in _RATES_HUB_HINTS):
        return PAGE_TYPE_RATES_HUB
    if any(h in u for h in _BRANCH_URL_HINTS):
        return PAGE_TYPE_BRANCH
    if any(h in u for h in _NEWS_URL_HINTS):
        return PAGE_TYPE_NEWS
    if "/learn/blog/" in u and "calculator" in u:
        return PAGE_TYPE_CALCULATOR
    if "/learn/blog/" in u:
        return PAGE_TYPE_NEWS
    if any(h in u for h in ("/calculators/", "switch-and-earn")):
        return PAGE_TYPE_CALCULATOR
    if "/personal/" in u or "/business/" in u:
        return PAGE_TYPE_PRODUCT
    return PAGE_TYPE_OTHER


BYLINE_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b"
)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"]
)}


def extract_article_date(markdown: str) -> Optional[str]:
    """Blog/newsroom pages open with a byline date ('June 20, 2025'). That date
    is when the article's claims were true, which can be years before the
    crawl that happened to re-fetch the same archived page."""
    head = markdown[:600]
    m = BYLINE_DATE_RE.search(head)
    if not m:
        return None
    month, day, year = m.group(1), int(m.group(2)), int(m.group(3))
    return f"{year:04d}-{MONTHS[month]:02d}-{day:02d}"


# --------------------------------------------------------------------------
# FactRow: one numeric value for one product/tier from one table row
# --------------------------------------------------------------------------

@dataclass
class FactRow:
    family: str                 # normalized product/section name, e.g. "auto & motorcycle loans"
    label: str                  # the row's own identifying text, e.g. "2024-27 Model Year"
    metric: str                 # "apr" | "apy" | "dividend_rate" | "rate" | "fee" | "amount"
    value_low: float
    value_high: Optional[float]
    qualifiers: dict            # term_months, model_year_lo/hi, balance_lo/hi, loan_amount_hi, ira_type, is_intro...
    url: str
    title: str
    page_type: str
    snapshot_id: str
    taken_on: str
    fetched_at: str
    effective_date: Optional[str]
    quote: str = ""             # short excerpt for citation
    raw_low: str = ""           # exact matched text, e.g. "5.39" or "0.20" — avoids reformatting drift
    raw_high: str = ""

    @property
    def is_business(self) -> bool:
        return "/business/" in self.url.lower() or "business" in self.family.lower() or "business" in self.label.lower()

    def display(self) -> str:
        unit = "%" if self.metric in ("apr", "apy", "dividend_rate", "rate") else ""
        suffix = {"apr": " APR", "apy": " APY", "dividend_rate": " dividend rate"}.get(self.metric, "")
        if unit == "%":
            if self.raw_high:
                return f"{self.raw_low}%-{self.raw_high}%{suffix}"
            return f"{self.raw_low}%{suffix}"
        if self.metric in ("fee", "min_to_open", "min_to_earn"):
            return f"${self.raw_low}"
        return self.raw_low


PCT_CELL_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%(?:\s*[-–—]\s*(\d+(?:\.\d+)?)\s*%)?\s*(APR|APY)?", re.I
)
DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
TERM_MONTHS_RE = re.compile(r"(\d+)\s*[-\s]?\s*(?:month|mo\b)", re.I)
TERM_YEARS_RE = re.compile(r"(\d+)\s*[-\s]?\s*year", re.I)
MODEL_YEAR_RANGE_RE = re.compile(r"(?:model year\s*)?(\d{4})\s*[-–—]\s*(\d{2,4})", re.I)
MODEL_YEAR_SINGLE_RE = re.compile(r"(\d{4})\s*model year", re.I)
BALANCE_RANGE_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*[-–—]\s*\$?\s*([\d,]+(?:\.\d+)?)"
)
BALANCE_PLUS_RE = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*\+")
LOAN_CEILING_RE = re.compile(r"\$\s*([\d,]+)\s*(?:or less)?")

METRIC_WORDS = {
    "apr": "apr",
    "annual percentage rate": "apr",
    "apy": "apy",
    "annual percentage yield": "apy",
    "dividend rate": "dividend_rate",
}


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _detect_metric(cell_text: str, header: str) -> Optional[str]:
    t = cell_text.lower()
    if "apr" in t:
        return "apr"
    if "apy" in t:
        return "apy"
    h = header.lower()
    for k, v in METRIC_WORDS.items():
        if k in h:
            return v
    if "dividend" in h:
        return "dividend_rate"
    if "%" in cell_text:
        return "rate"
    return None


def _extract_qualifiers(row_cells: list) -> dict:
    """Scan every cell's text (ignoring header names) for tier signals."""
    q: dict = {}
    blob = " | ".join(v for _, v in row_cells)
    if m := TERM_MONTHS_RE.search(blob):
        q["term_months"] = int(m.group(1))
    elif m := TERM_YEARS_RE.search(blob):
        q["term_months"] = int(m.group(1)) * 12
    if m := MODEL_YEAR_RANGE_RE.search(blob):
        lo = int(m.group(1))
        hi_raw = m.group(2)
        hi = int(hi_raw) if len(hi_raw) == 4 else (lo // 100) * 100 + int(hi_raw)
        if hi < lo:  # e.g. "2026-23" means 2023-2026
            lo, hi = hi, lo
        q["model_year_lo"], q["model_year_hi"] = lo, hi
    elif m := MODEL_YEAR_SINGLE_RE.search(blob):
        q["model_year_lo"] = q["model_year_hi"] = int(m.group(1))
    elif "model year (and older)" in blob.lower():
        m2 = re.search(r"(\d{4})\s*model year", blob, re.I)
        if m2:
            q["model_year_hi"] = int(m2.group(1))
            q["model_year_and_older"] = True
    for header, val in row_cells:
        hl = header.lower()
        if "balance" in hl or "loan amount" in hl:
            if bm := BALANCE_RANGE_RE.search(val):
                q["amount_lo"] = _num(bm.group(1))
                q["amount_hi"] = _num(bm.group(2))
            elif pm := BALANCE_PLUS_RE.search(val):
                q["amount_lo"] = _num(pm.group(1))
                q["amount_hi"] = None
        if "or less" in val.lower():
            if lm := LOAN_CEILING_RE.search(val):
                q["amount_hi"] = _num(lm.group(1))
                q["is_jumbo"] = q.get("amount_hi", 0) and q["amount_hi"] > 400000
    if re.search(r"\bjumbo\b", blob, re.I):
        q["is_jumbo"] = True
    if re.search(r"\bintro(ductory)?\b", blob, re.I):
        q["is_intro"] = True
    if m := re.search(r"\broth\b", blob, re.I):
        q["ira_type"] = "roth"
    elif re.search(r"\btraditional\b", blob, re.I):
        q["ira_type"] = "traditional"
    return q


def _clean_family(section: str) -> str:
    s = section.strip().rstrip(":").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def rows_from_page(url: str, title: str, markdown: str, page_type: str,
                    snapshot_id: str, taken_on: str, fetched_at: str) -> list:
    facts = []
    for table in extract_tables(markdown):
        if not table.rows:
            continue
        # Gate: only treat this as a *rate* table if the site stamped it with
        # an effective date, OR it is plainly a fee table ($ values, "fee" in
        # the header/section). This is what filters out amortization-illustration
        # tables, vehicle-for-sale listings, and empty widget skeletons.
        any_effective = any(r.effective_date for r in table.rows)
        header_blob = " ".join(table.headers).lower()
        looks_like_fee_table = "fee" in header_blob and any(
            DOLLAR_RE.search(v) for r in table.rows for v in r.cells.values()
        )
        if not any_effective and not looks_like_fee_table:
            continue
        for row in table.rows:
            qualifiers = _extract_qualifiers(row.cell_list)
            label_cell = next((v for h, v in row.cell_list if v and not PCT_CELL_RE.search(v)
                                and not DOLLAR_RE.fullmatch(v.strip())), "")
            family = _clean_family(row.section)
            for header, val in row.cell_list:
                if not val:
                    continue
                metric = _detect_metric(val, header)
                if metric is None:
                    continue
                m = PCT_CELL_RE.search(val)
                if not m:
                    continue
                lo = float(m.group(1))
                hi = float(m.group(2)) if m.group(2) else None
                # A 0% loan rate is never real — it's the "Points" column (0)
                # bleeding into a rate cell. Deposit metrics keep 0.00%
                # (Free Checking legitimately pays 0.00% APY).
                if metric in ("apr", "rate") and lo == 0:
                    continue
                facts.append(FactRow(
                    family=family,
                    label=label_cell or row.cells.get(table.headers[0], ""),
                    metric=metric,
                    value_low=lo,
                    value_high=hi,
                    raw_low=m.group(1),
                    raw_high=m.group(2) or "",
                    qualifiers=qualifiers,
                    url=url,
                    title=title,
                    page_type=page_type,
                    snapshot_id=snapshot_id,
                    taken_on=taken_on,
                    fetched_at=fetched_at,
                    effective_date=row.effective_date,
                    quote=row.raw_line.strip(),
                ))
            # "Min. to Open" / "Min. to Earn" columns on deposit tables are
            # structured facts too, so "how much to open X?" doesn't have to
            # fall back to prose retrieval.
            for header, val in row.cell_list:
                hl = header.lower()
                if "min" not in hl or "to open" not in hl and "to earn" not in hl:
                    continue
                dm = DOLLAR_RE.search(val)
                if not dm:
                    continue
                facts.append(FactRow(
                    family=family, label=label_cell or row.cells.get(table.headers[0], ""),
                    metric="min_to_open" if "to open" in hl else "min_to_earn",
                    value_low=_num(dm.group(1)), value_high=None,
                    raw_low=dm.group(1), raw_high="",
                    qualifiers=qualifiers, url=url, title=title, page_type=page_type,
                    snapshot_id=snapshot_id, taken_on=taken_on, fetched_at=fetched_at,
                    effective_date=row.effective_date, quote=row.raw_line.strip(),
                ))
            if looks_like_fee_table:
                for header, val in row.cell_list:
                    dm = DOLLAR_RE.search(val)
                    if dm and header != label_cell:
                        facts.append(FactRow(
                            family=family or "fees",
                            label=label_cell,
                            metric="fee",
                            value_low=_num(dm.group(1)),
                            value_high=None,
                            raw_low=dm.group(1),
                            raw_high="",
                            qualifiers=qualifiers,
                            url=url, title=title, page_type=page_type,
                            snapshot_id=snapshot_id, taken_on=taken_on, fetched_at=fetched_at,
                            effective_date=row.effective_date,
                            quote=row.raw_line.strip(),
                        ))
    facts.extend(_inline_rate_rows(url, title, markdown, page_type,
                                   snapshot_id, taken_on, fetched_at))
    return facts


# Inline prose rates: "6-month certificate APY: 3.50%". Strict on purpose so
# it doesn't fire on marketing copy — it requires a term/qualifier, a product
# word, an explicit APR/APY/rate metric, and a single percentage. Sites that
# publish rates as sentences instead of tables
# are still covered without polluting the table-based dupaco answers.
INLINE_RATE_RE = re.compile(
    r"([A-Za-z0-9][\w\-\s]{0,40}?)\s+(APR|APY|rate)\s*[:=]?\s*(?:is\s+|of\s+)?(\d+(?:\.\d+)?)\s*%",
    re.I,
)
INLINE_PRODUCT_WORDS = re.compile(
    r"certificate|\bcd\b|savings|checking|loan|mortgage|ira|money market|visa|heloc", re.I
)


def _inline_rate_rows(url, title, markdown, page_type, snapshot_id, taken_on, fetched_at) -> list:
    out = []
    article_date = extract_article_date(markdown) if page_type in (PAGE_TYPE_NEWS, PAGE_TYPE_CALCULATOR) else None
    for line in markdown.split("\n"):
        if line.lstrip().startswith("|"):
            continue
        if re.search(r"offer ended|expired|no longer|ended \d{4}-\d{2}-\d{2}", line, re.I):
            continue  # an explicitly-expired promo line is never a current rate
        for m in INLINE_RATE_RE.finditer(line):
            label, metric_word, value = m.group(1).strip(), m.group(2).lower(), m.group(3)
            if not INLINE_PRODUCT_WORDS.search(label):
                continue
            metric = {"apr": "apr", "apy": "apy", "rate": "rate"}[metric_word]
            qualifiers = _extract_qualifiers([("label", label)])
            out.append(FactRow(
                family=_clean_family(label), label=label, metric=metric,
                value_low=float(value), value_high=None,
                qualifiers=qualifiers, url=url, title=title, page_type=page_type,
                snapshot_id=snapshot_id, taken_on=taken_on, fetched_at=fetched_at,
                effective_date=article_date, quote=line.strip()[:200],
                raw_low=value, raw_high="",
            ))
    return out


# --------------------------------------------------------------------------
# EntityFact: non-numeric facts pulled from free text
# --------------------------------------------------------------------------

@dataclass
class EntityFact:
    category: str        # "routing_number" | "phone" | "url" | "branch_count" | "address" | "hours" | "text"
    key: str              # e.g. "routing_number", "galena_address", "hy-vee_closure"
    value: str
    url: str
    title: str
    page_type: str
    snapshot_id: str
    taken_on: str
    fetched_at: str
    effective_date: Optional[str]   # article date if this came from a dated article
    quote: str = ""


# Routing/ABA number in any of its on-site phrasings: "Routing Number: 2739...",
# "routing number (2739...)", "ABA number (2739...)".
ROUTING_RE = re.compile(
    r"(?:routing|aba)\s*(?:number|#)?\s*[:(]?\s*(\d{9})\b"
    r"|(\d{9})\b(?=[^.]{0,30}(?:routing|aba))",
    re.I,
)
PHONE_RE = re.compile(r"\b(800-373-7600)\b")
MORTGAGE_PHONE_RE = re.compile(r"Mortgage Team at\s*1?[.\s]?(800[.\-]383[.\-]1700)", re.I)
BRANCH_COUNT_RE = re.compile(r"(\d+)\s*branch office location")
ADDRESS_LINE_RE = re.compile(r"^(?P<street>[\w.&'’\- ]+(?:Rd|Rd\.|St|St\.|Ave|Ave\.|Ln|Ln\.|Dr|Dr\.|Blvd|Blvd\.|Way)\.?,\s*[A-Za-z .]+,\s*(?:IA|IL|WI),\s*\d{5})\s*$")


def entity_facts_from_page(url: str, title: str, markdown: str, page_type: str,
                            snapshot_id: str, taken_on: str, fetched_at: str) -> list:
    out = []
    article_date = extract_article_date(markdown) if page_type == PAGE_TYPE_NEWS else None
    eff = article_date  # for news/blog pages, the article's own date is its "effective date"

    def add(category, key, value, quote=""):
        out.append(EntityFact(category, key, value, url, title, page_type,
                               snapshot_id, taken_on, fetched_at, eff, quote))

    for m in ROUTING_RE.finditer(markdown):
        add("routing_number", "routing_number", m.group(1) or m.group(2), m.group(0))
    for m in PHONE_RE.finditer(markdown):
        add("phone", "main_phone", m.group(1))
    for m in MORTGAGE_PHONE_RE.finditer(markdown):
        add("phone", "mortgage_phone", m.group(1).replace(".", "-"), m.group(0))
    for m in BRANCH_COUNT_RE.finditer(markdown):
        add("branch_count", "branch_count", m.group(1), m.group(0))
    for m in re.finditer(r"(first|second|third|fourth|fifth|sixth|seventh)\s+time\b", markdown, re.I):
        window = markdown[max(0, m.start() - 80):m.end() + 80]
        if re.search(r"forbes", window, re.I):
            add("forbes_recognition", "forbes_nth", m.group(1).lower(), window.strip())
    for m in re.finditer(r"close[sd]?\s+(?:on\s+)?([A-Z][a-z]+ \d{1,2},\s*\d{4})", markdown):
        add("branch_closure_date", "hy-vee_closure" if "hy-vee" in markdown.lower()[:2000] else "closure", m.group(1), m.group(0))
    return out
