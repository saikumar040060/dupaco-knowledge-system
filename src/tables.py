"""
Generic markdown-table extractor.

The site's pages are HubSpot/CMS exports: every rate, dividend and fee table
renders as a GitHub-style pipe table, but the *header text* on a table is not
reliable (the source data deliberately demonstrates this — a vehicle calculator
page labels an auto-loan APR table "Balance Tier | Annual Percentage Yield").
So this module treats headers as a hint only. The ground truth for "what kind
of number is this" comes from the text actually inside each cell (a trailing
"APR"/"APY"/"Dividend Rate" token, or a %, or a $ amount), and the ground
truth for "which product/tier is this row about" comes from scanning every
cell for recognizable qualifier patterns (terms, model-year ranges, balance
ranges, loan-amount ceilings, IRA type, intro-vs-standard wording) rather than
trusting a single column position.

Each extracted row keeps the section heading that introduced its table and
the "Rates are effective as of <date>" sentence that follows it (these pages
put the effective-date footer directly under each table, not once per page),
because `policy.py` needs a per-table, not per-page, effective date.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

PIPE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
SEP_CELL = re.compile(r"^\s*:?-{2,}:?\s*$")
EFFECTIVE_DATE_RE = re.compile(
    r"effective\s+as\s+of\s+(\d{1,2})/(\d{1,2})/(\d{4})", re.I
)
EFFECTIVE_DATE_RE2 = re.compile(
    r"effective\s+(\d{1,2})/(\d{1,2})/(\d{4})", re.I
)
HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")
FOOTNOTE_SUFFIX = re.compile(r"(?<=[%\d])\s?[¹²³⁴-⁹]|(?<=%)(?=\d\b)")


@dataclass
class TableRow:
    cells: dict  # header -> raw cell text (header may be duplicated/blank)
    cell_list: list  # ordered (header, text) pairs, for header-agnostic scans
    section: str  # nearest heading/label text above the table
    effective_date: Optional[str]  # ISO date string, or None
    raw_line: str


@dataclass
class Table:
    headers: list
    rows: list = field(default_factory=list)


def _clean_cell(text: str) -> str:
    text = text.strip()
    text = text.replace("\\*", "*").replace("\\r\\n", " ")
    # Footnote reference digits glue onto numbers with no space, e.g. "5.39% APR1"
    # or "4.20%1" or "15.50%1". Strip a single trailing digit that immediately
    # follows a unit token, but only when it isn't itself part of a longer number.
    text = re.sub(r"(%|APR|APY)(\d)(?!\d)", r"\1", text)
    return text.strip()


def _find_section(lines: list, idx: int) -> str:
    """Nearest non-empty line above `idx` that looks like a label, not markup.
    These pages often pad a section label with many blank lines before the
    table, so the budget below counts non-blank lines examined, not raw
    line distance."""
    examined = 0
    j = idx - 1
    while j >= 0 and examined < 8:
        line = lines[j].strip()
        j -= 1
        if not line:
            continue
        examined += 1
        m = HEADING_RE.match(line)
        if m:
            return m.group(1)
        if line.startswith("|") or line.startswith("["):
            continue
        if len(line) < 80:
            return line
    return ""


def _find_effective_date(lines: list, idx: int) -> Optional[str]:
    """Scan after the table for its effective-date footer. Same reasoning as
    `_find_section`: budget non-blank lines, since the footer is often many
    blank lines below the table's last row."""
    examined = 0
    j = idx
    while j < len(lines) and examined < 6:
        line = lines[j]
        j += 1
        if not line.strip():
            continue
        examined += 1
        m = EFFECTIVE_DATE_RE.search(line) or EFFECTIVE_DATE_RE2.search(line)
        if m:
            mm, dd, yyyy = m.groups()
            try:
                return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
            except ValueError:
                continue
        if line.strip().startswith("|") or HEADING_RE.match(line.strip()):
            break
    return None


def extract_tables(markdown: str) -> list:
    """Return every pipe table in `markdown` as a list of Table objects."""
    lines = markdown.split("\n")
    tables = []
    i = 0
    n = len(lines)
    while i < n:
        m = PIPE_ROW.match(lines[i])
        if not m:
            i += 1
            continue
        header_cells = [_clean_cell(c) for c in m.group(1).split("|")]
        # need a separator row right after to confirm this is a real table
        if i + 1 >= n:
            i += 1
            continue
        sep_m = PIPE_ROW.match(lines[i + 1])
        if not sep_m or not all(SEP_CELL.match(c) for c in sep_m.group(1).split("|")):
            i += 1
            continue
        section = _find_section(lines, i)
        body_start = i + 2
        j = body_start
        data_rows = []
        while j < n:
            rm = PIPE_ROW.match(lines[j])
            if not rm:
                break
            cells = [_clean_cell(c) for c in rm.group(1).split("|")]
            data_rows.append((lines[j], cells))
            j += 1
        effective_date = _find_effective_date(lines, j)
        table = Table(headers=header_cells)
        # de-duplicate blank/duplicate headers by position so cell_list stays usable
        for raw_line, cells in data_rows:
            width = max(len(header_cells), len(cells))
            padded_headers = header_cells + [""] * (width - len(header_cells))
            padded_cells = cells + [""] * (width - len(cells))
            cell_list = list(zip(padded_headers, padded_cells))
            cell_dict = {}
            for h, v in cell_list:
                if h and h not in cell_dict:
                    cell_dict[h] = v
            table.rows.append(
                TableRow(
                    cells=cell_dict,
                    cell_list=cell_list,
                    section=section,
                    effective_date=effective_date,
                    raw_line=raw_line,
                )
            )
        if table.rows:
            tables.append(table)
        i = j
    return tables
