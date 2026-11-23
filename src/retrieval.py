"""
Deterministic BM25 retrieval over paragraph-level chunks of a snapshot's
pages, used only as a fallback when qa.py can't route a question to a
structured fact lookup (prose answers: branch info, eligibility, product
descriptions, etc.).

Built once per snapshot at ingest time (ingest is off the answer path);
queried at answer time, which must stay fast and must give the same chunk
ranking every time for the same question (consistency scoring depends on
this), so there is no randomness anywhere in this module.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

WORD_RE = re.compile(r"[a-z0-9$%']+")

STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "what", "which", "how",
    "i", "you", "your", "my", "to", "of", "for", "and", "or", "in", "on",
    "can", "will", "be", "it", "at", "with", "this", "that", "me", "please",
}

# A small set of site-specific synonyms the data exercises
# (CD/certificate, Home$tart, etc.) plus common phrasing variants, expanded
# into the query so a paraphrase retrieves the same chunks as the original.
SYNONYMS = {
    "cd": ["certificate"],
    "cds": ["certificates"],
    "certificates": ["certificate"],
    "apy": ["apy", "dividend", "rate"],
    "hometart": ["homestart"],
    "home$tart": ["homestart"],
    "routing": ["routing", "aba"],
    "mortgage": ["mortgage", "home", "loan"],
    "atm": ["atm", "locator", "branch"],
    "branches": ["branch"],
    "locations": ["branch", "location"],
    "hours": ["hours", "open", "closed"],
}


def tokenize(text: str) -> list:
    return [t for t in WORD_RE.findall(text.lower()) if t not in STOPWORDS]


def expand(tokens: list) -> list:
    out = list(tokens)
    for t in tokens:
        out.extend(SYNONYMS.get(t, []))
    return out


@dataclass
class Chunk:
    chunk_id: int
    url: str
    title: str
    page_type: str
    text: str


class BM25Index:
    """One per snapshot."""

    K1 = 1.5
    B = 0.75

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.doc_freq: dict = defaultdict(int)       # term -> # chunks containing it
        self.postings: dict = defaultdict(list)       # term -> [(chunk_id, tf)]
        self.doc_len: list = []
        self.avg_len = 0.0

    def add_page(self, url: str, title: str, page_type: str, markdown: str) -> None:
        for para in _paragraphs(markdown):
            if len(para) < 20:
                continue
            cid = len(self.chunks)
            self.chunks.append(Chunk(cid, url, title, page_type, para))
            toks = tokenize(para)
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            for term, count in tf.items():
                self.doc_freq[term] += 1
                self.postings[term].append((cid, count))

    def finalize(self) -> None:
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0

    def search(self, query: str, top_k: int = 8) -> list:
        tokens = expand(tokenize(query))
        if not tokens or not self.chunks:
            return []
        n_docs = len(self.chunks)
        scores: dict = defaultdict(float)
        qtf = Counter(tokens)
        for term, qf in qtf.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for cid, tf in postings:
                dl = self.doc_len[cid] or 1
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self.avg_len)
                scores[cid] += idf * (tf * (self.K1 + 1)) / denom * qf

        qtoks_set = set(tokens)
        for cid in list(scores.keys()):
            chunk = self.chunks[cid]
            title_hit = len(qtoks_set & set(tokenize(chunk.title)))
            url_hit = len(qtoks_set & set(tokenize(chunk.url.replace("-", " ").replace("/", " "))))
            scores[cid] += 0.6 * title_hit + 0.3 * url_hit

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [(self.chunks[cid], score) for cid, score in ranked]


def _paragraphs(markdown: str) -> list:
    paras = []
    for block in re.split(r"\n\s*\n", markdown):
        block = block.strip()
        if not block:
            continue
        if block.startswith("|"):
            # A table. It is NOT turned into a rate fact here (facts.py decides
            # that from the "effective as of" date / fee header); but its text —
            # feature comparisons, branch hours, etc. — should still be findable
            # by the prose fallback, so flatten each row into a readable line.
            flat = _flatten_table(block)
            if flat:
                paras.append(flat)
            continue
        paras.append(re.sub(r"\s+", " ", block))
    return paras


_SEP_ROW = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _flatten_table(block: str) -> str:
    """Turn a pipe table into one searchable paragraph: drop the separator row,
    pair each data cell with its column header ('Lobby hours: 9 am-5 pm')."""
    rows = []
    for line in block.split("\n"):
        line = line.strip()
        if not line.startswith("|") or _SEP_ROW.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return ""
    headers = rows[0]
    out = [" ".join(headers)]
    for cells in rows[1:]:
        pairs = []
        for i, c in enumerate(cells):
            if not c:
                continue
            h = headers[i] if i < len(headers) and headers[i] else ""
            pairs.append(f"{h}: {c}" if h else c)
        if pairs:
            out.append("; ".join(pairs))
    return re.sub(r"\s+", " ", " . ".join(out))
