# Dupaco Knowledge System

A deterministic, dependency-free knowledge system that reads **versioned
snapshots of a website** (the same site crawled repeatedly over months) and
answers questions the way a careful human expert would: with the **current**
value, the **right tier**, a **source citation**, and an honest *"I don't
know"* when the site doesn't actually say.

Websites lie by accident. A rate changes but the old page is never taken down.
A promotion ends but its landing page lingers. One page quotes a *rate* and
another an *APY* and they look like they disagree. A table has tiers (by
balance, by term, by model year) and read carelessly every tier looks like a
conflict. This system is built to be right anyway.

It uses no network, no LLM, and no randomness on the answer path, so every
answer is reproducible from the source data alone and every paraphrase of a
question returns the identical set of numbers.

## Quick start

```bash
# No dependencies to install — the system is pure Python 3.11 standard library.
python -c "import sys; sys.path.insert(0,'src'); import solution; print(solution.build())"
```

```python
import sys; sys.path.insert(0, "src")
import solution
from contract import Snapshot, Page

ks = solution.build()
for snap in snapshots_oldest_to_newest:      # each is a Snapshot of Page objects
    ks.ingest(snap)

ans = ks.answer("What's the current rate on a 15-year fixed mortgage?")
print(ans.text, ans.confident, [s.url for s in ans.sources])

for c in ks.conflicts():
    print(c.topic, c.values, "->", c.resolution)
```

The boundary types (`Snapshot`, `Page`, `Answer`, `Source`, `Conflict`,
`KnowledgeSystem`) live in [`src/contract.py`](src/contract.py).

## How it works

Ingest does the heavy lifting; answering is deliberately cheap. For every page
in every snapshot the system records where it came from, when it was crawled,
the page's own "effective as of" date, and a **page-type** label derived from
its URL and title (official rates hub / product page / promo-or-calculator /
blog-news-press / branch locator / other). Every markdown table is parsed into
structured **FactRows** — one value for one product, one tier, one metric — and
free-text entities (routing number, phone, branch count, hours, addresses)
become **EntityFacts**. History is never overwritten: every snapshot's facts
coexist in one versioned store.

At answer time the question is parsed, routed to a structured lookup first and a
BM25 prose fallback second, and the competing candidates are resolved through a
single written-down **"what's current" policy**:

1. **Latest snapshot that still contains the fact** — recency of the *pull*
   dominates, so a stale page can't outvote a current one just by looking
   authoritative.
2. **Page authority** — the rates hub / product page beats a promo or
   calculator, which beats a blog or press release.
3. **Newest table "effective as of" date**, then **newest crawl timestamp**.
4. On a full tie, cite the product's own page.

`as_of` questions drop every snapshot taken after the given date, then apply the
same policy to what remains — so a question about a special "in May 2026"
returns the rate that was true then, not what it later became.

`conflicts()` re-scans the store and emits **one entry per (product, tier,
metric)** where current-looking pages genuinely disagree, listing only the
disagreeing values with a dated reason for the winner.

See **[DESIGN.md](DESIGN.md)** for the full write-up: tier normalization,
refusal logic, conflict precision, and known limitations.

## Layout

| Path | Role |
|---|---|
| `src/solution.py` | `build()` entry point; wires ingest → answer → conflicts |
| `src/ingest.py` | per-snapshot orchestration; versioned fact store + BM25 index |
| `src/tables.py` | markdown-table parser (headers are a hint, cells are truth) |
| `src/facts.py` | FactRow / EntityFact extraction, page classification, tier normalization |
| `src/policy.py` | the single "what's current" precedence policy |
| `src/retrieval.py` | deterministic BM25 fallback for prose questions |
| `src/qa.py` | question understanding, routing, deterministic answer composition |
| `src/conflicts.py` | precision-first contradiction detection |
| `src/contract.py` | the boundary data model |
| `tests/` | ~80 tests (unit + end-to-end) |
| `dev/` | hand-written evaluation fixtures |

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

The end-to-end tests run against a corpus of dated snapshots placed in a
`snapshots/` directory; without that data they **skip** cleanly (the pure-logic
tests — table parsing, tier parsing, policy, routing — always run). The crawl
corpus used during development is private and is not distributed with this
repository.

## Design principles

- **Deterministic.** No model calls, no embeddings, no dict-ordering surprises.
  The same question — and any paraphrase that parses to the same
  product/tier/metric — yields the identical answer every time.
- **Honest.** If a product, term, or tier simply isn't on the site, the system
  returns `confident=False` with no stray numbers rather than guessing.
- **Explainable.** Every "current" decision follows one short written-down
  policy, and every answer carries its source.
