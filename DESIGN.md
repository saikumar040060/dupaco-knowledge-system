# Design notes — Dupaco Knowledge System

## Architecture in one paragraph

Ingest does all the heavy lifting (it isn't timed; answering is). For every
page in every snapshot the system records where it came from, when it was
crawled, the page's own "effective as of" date, and a **page-type** label
derived from its URL and title (official rates hub / product page /
promo-or-calculator / blog-news-press / branch locator / other). Every markdown
table is parsed into structured **FactRows** — one numeric value for one
product, one tier, one metric — and free-text entities (routing number, phone,
branch count, hours, addresses) become **EntityFacts**. History is never
overwritten: all snapshots' facts coexist in one store. At answer time the
question is parsed, routed to a structured lookup first and a BM25 prose
fallback second, the competing candidates are resolved through a single
written-down "current" policy, and a deterministic, template-based answer is
composed with the winning fact's source URL. `conflicts()` re-scans the same
store for genuine same-snapshot contradictions.

## How "current" is decided

All in `policy.py`, in this order:

1. **Latest snapshot that still contains the fact.** Recency of the *pull*
   dominates — a page that disappeared or changed by the newest crawl must not
   be outvoted by an authoritative-looking page that is simply stale.
2. **Page authority.** Within the newest snapshot: the `/rates` hub and a
   product's own page beat a promo/calculator page, which beats a blog or
   press release. This is what makes a stale calculator page (still showing last
   quarter's auto rate) lose to the rates page's current value.
3. **Newest table "effective as of" date.** The deposit and loan halves of a
   `/rates` page are stamped on different days; the newer stamp wins.
4. **Newest crawl timestamp**, as a final tie-break.
5. A citation-only preference: on a full tie, cite the product's own page.

For `as_of` questions the system first drops every snapshot taken after the
date, then applies the same policy to what remains — so a question about a
special "in May 2026" returns the value that was true then, not what it later
became.

## Tiers and qualifiers

Tiers are the heart of the trap set, so a FactRow carries normalized qualifiers
scanned from *all* of a row's cells (never a trusted column position, because
the data deliberately ships mislabeled headers — a calculator labels an auto-APR
table "Annual Percentage Yield"):

- **term** ("60 Months", "30-Year" → 360, "120 mo"),
- **model-year range** — normalized so the label drift `2023-26 → 2024-27` still
  maps to the same "newest vehicles" tier,
- **balance / loan-amount range** and **loan-amount ceiling** ("$832,750 or
  less"),
- **jumbo**, **intro** (a balance-transfer teaser is dropped so it never
  masquerades as the standard card APR), **Roth/Traditional**.

When a question names a tier, the system answers exactly that tier. When it
doesn't, each product family has a convention: balance products quote the
headline tier; mortgages lead with the $100k loan-amount tier (plus the jumbo
ceiling), never the misleading headline; model-year products quote the newest
tier; certificates read back every term. Rate vs APY vs dividend-rate are kept
distinct and chosen from the question.

## Refusals

If a product/term/tier simply isn't on the site — a 7/1 ARM, a construction
loan, a 90-month auto term, a branch that doesn't publish hours — the system
returns `confident=False` with "I don't know … please contact …" and **no**
stray numbers. This is driven structurally: a rate-shaped question that routes
to a product bucket but finds no matching tier refuses rather than guessing, a
request for a rate *above* the largest published tier ceiling refuses instead of
quoting the top tier, an ungrounded promotional offer (a specific bonus the site
doesn't state) abstains, and a private/CEO phone number abstains while the
general line still answers.

## Conflict detection (precision-first)

`conflicts()` emits **one entry per (product, tier, metric)** where two or more
current-looking pages disagree, listing only the disagreeing values with a dated
reason for the winner. Every entry is therefore individually defensible; there
is deliberately no bundling that would mix unrelated tiers under one "winner".

Deliberately *not* conflicts: jumbo vs standard, Roth vs Traditional, one
balance/term/model-year tier vs another (folded into the tier key); values that
changed cleanly over time (compared only within the latest snapshot); and
equal-authority, equal-date differences (treated as rounding/parse noise — a
real conflict needs a demonstrably stale loser). Model-year label drift is
handled by clustering the tiers' upper bounds within a year of each other
(`2023-26` ≡ `2024-27`), which never pairs the newest tier on one page with an
older tier on a page that lists fewer tiers for that term.

## What's deterministic, and why

Everything on the answer path. No model calls, no embeddings with random
initialization, no dict-ordering surprises (ties broken by explicit keys). The
same question — and any paraphrase that parses to the same product/tier/metric —
yields the identical set of numbers every time. Measured p95 answer latency is
~0.014s.

## Why no LLM at answer time

The LLM is deliberately kept off the answer path: letting a model choose the
number at answer time would break determinism and risk hallucinated rates. If
one were used at all, it would be an *optional*, env-gated ingest-time aid for
messy prose extraction, with the committed rule-based extractor as the always-on
fallback — but the rule-based extractor already covers the data, so it isn't
wired in.

## Production-shaped storage

Although a fresh process rebuilds the store from the snapshots on startup, the
store is written the way a real deployment would want it: each page is
fingerprinted (SHA-1) so an unchanged page on a later crawl is not re-parsed,
and each distinct fact/entity value is interned once with a compact list of the
(snapshot, dates, sequence) sightings that carry it, rather than stored per
snapshot. In a real system the interned units would live in a database or object
store; here they live in memory, but the access pattern is identical.

## Known limitations

- **Prose fallback grounding is heuristic.** Free-text answers require the
  question's distinctive terms (after removing stop words and generic banking
  vocabulary) to co-occur with the product noun in the matched passage, and
  blogs/promos are used only for *known* products. This makes "do you have a
  crypto debit card?" abstain while "does the Rewards Visa charge an annual
  fee?" still answers from prose. The risk is two-sided and mitigated, not
  eliminated.
- **Conflict scope is within the latest snapshot only.** A value that was
  cleanly replaced everywhere between snapshots is intentionally not reported.
- **Model-year parsing** reads only a 4-digit year (1990–2039) adjacent to a
  vehicle word (or after "a / my / new / used") as a model year; otherwise the
  system lists all model-year tiers. "Brand new car" with no year resolves to
  the newest tier.
- **Branch-specific hours/addresses** rely on the branch directory and
  new-branch announcements; a branch that publishes neither correctly refuses
  rather than guessing.
- **Non-rate tables** (feature comparisons, hours grids) are not turned into
  rate facts — only dated rate tables and fee tables are, so payment-example
  tables can't leak fake numbers. They are, however, flattened into the prose
  search index (header paired with each cell) so their content is findable.
