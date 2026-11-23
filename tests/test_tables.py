"""Table parser: the header text is untrusted, the cell content is truth."""
from tables import extract_tables


def test_parses_basic_rate_table_with_effective_date():
    md = """
## Auto & Motorcycle Loans

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2024-27 Model Year | 60 Months | 5.39% APR1 |

Rates are effective as of 08/31/2026. All loans are subject to approval.
"""
    tables = extract_tables(md)
    assert len(tables) == 1
    row = tables[0].rows[0]
    assert row.section == "Auto & Motorcycle Loans"
    assert row.effective_date == "2026-08-31"
    # footnote digit stripped off the value
    assert row.cell_list[-1][1] == "5.39% APR"


def test_effective_date_survives_blank_line_padding():
    # The CMS export pads many blank lines between the last row and the footer.
    md = """
Special

| Term | APY |
| --- | --- |
| 10 Month Fixed Jumbo Certificate Special | 3.90% |




**Rates are effective as of 03/03/2026 and are subject to change.**
"""
    tables = extract_tables(md)
    assert tables[0].rows[0].effective_date == "2026-03-03"


def test_section_label_found_across_blank_lines():
    md = """
Share Savings



| Account Type | Balance Tier | APY |
| --- | --- | --- |
| Share Savings | $25,000+ | 0.20% |

Rates are effective as of 06/15/2026.
"""
    assert extract_tables(md)[0].rows[0].section == "Share Savings"


def test_mislabeled_header_does_not_change_cell_contents():
    # A calculator page labels an auto APR table "Annual Percentage Yield".
    md = """
Auto Loan Rates

| Account Type | Balance Tier | Annual Percentage Yield |
| --- | --- | --- |
| 2023-26 Model Year | 60 Months | 5.34% APR |

Rates are effective as of 1/1/2026.
"""
    row = extract_tables(md)[0].rows[0]
    # the wrong header is preserved verbatim; the cell still says APR
    assert "Annual Percentage Yield" in row.cells
    assert row.cell_list[-1][1] == "5.34% APR"


def test_non_table_text_is_ignored():
    assert extract_tables("# Heading\n\nJust a paragraph, no pipes.") == []


def test_undated_table_is_flattened_for_search():
    """Non-rate tables (no 'effective as of' date) must become searchable text
    — paired with headers — instead of being dropped by the prose index."""
    from retrieval import _flatten_table, _paragraphs
    feat = ("| Feature | Free Checking |\n| --- | --- |\n"
            "| Monthly fee | None |\n| Online banking | Yes |")
    flat = _flatten_table(feat)
    assert "Monthly fee" in flat and "None" in flat and "Online banking" in flat
    assert "---" not in flat                      # separator row dropped
    # and such a table is no longer skipped by the paragraph splitter
    md = "Intro text.\n\n" + feat
    paras = _paragraphs(md)
    assert any("Monthly fee" in p for p in paras)


def test_hours_table_header_paired_with_value():
    from retrieval import _flatten_table
    flat = _flatten_table("| Branch | Lobby hours |\n| --- | --- |\n| Asbury Road | 9 am-5 pm |")
    assert "Lobby hours: 9 am-5 pm" in flat and "Branch: Asbury Road" in flat
