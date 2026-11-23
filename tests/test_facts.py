"""Fact extraction: qualifiers, page classification, tier normalization."""
from facts import (
    AUTHORITY_RANK,
    PAGE_TYPE_CALCULATOR,
    PAGE_TYPE_NEWS,
    PAGE_TYPE_PRODUCT,
    PAGE_TYPE_RATES_HUB,
    _extract_qualifiers,
    classify_page,
    extract_article_date,
    rows_from_page,
)


def test_classify_page_authority_order():
    assert classify_page("https://www.dupaco.com/rates", "Rates") == PAGE_TYPE_RATES_HUB
    assert classify_page("https://www.dupaco.com/personal/loans/auto-motorcycle-loans", "") == PAGE_TYPE_PRODUCT
    assert classify_page("https://www.dupaco.com/switch-and-earn", "") == PAGE_TYPE_CALCULATOR
    assert classify_page("https://www.dupaco.com/newsroom/x", "") == PAGE_TYPE_NEWS
    # authority is strictly ordered: hub/product beat calculator beat news
    assert AUTHORITY_RANK[PAGE_TYPE_RATES_HUB] < AUTHORITY_RANK[PAGE_TYPE_CALCULATOR]
    assert AUTHORITY_RANK[PAGE_TYPE_CALCULATOR] < AUTHORITY_RANK[PAGE_TYPE_NEWS]


def test_promo_pages_demoted_to_calculator_authority():
    assert classify_page("https://www.dupaco.com/summer-promo", "") == PAGE_TYPE_CALCULATOR


def test_model_year_range_normalized_regardless_of_label_drift():
    q1 = _extract_qualifiers([("Loan Program", "2024-27 Model Year"), ("Term", "60 Months")])
    q2 = _extract_qualifiers([("Loan Program", "2023–26 Model Year"), ("Term", "60 Months")])
    assert q1["model_year_lo"] == 2024 and q1["model_year_hi"] == 2027
    assert q2["model_year_lo"] == 2023 and q2["model_year_hi"] == 2026
    assert q1["term_months"] == q2["term_months"] == 60


def test_reversed_model_year_range():
    q = _extract_qualifiers([("Loan Type", "Model Year 2026-23")])
    assert q["model_year_lo"] == 2023 and q["model_year_hi"] == 2026


def test_term_parsing_handles_hyphen_and_mo():
    assert _extract_qualifiers([("x", "6-month certificate")])["term_months"] == 6
    assert _extract_qualifiers([("x", "Loan – Standard (120 mo)")])["term_months"] == 120
    assert _extract_qualifiers([("x", "30-Year Fixed")])["term_months"] == 360


def test_jumbo_and_amount_ceiling_detected():
    q = _extract_qualifiers([("Program", "30-Year Fixed Conventional $832,750 or less")])
    assert q["amount_hi"] == 832750.0
    assert q.get("is_jumbo") is True


def test_article_date_from_byline():
    assert extract_article_date("2 min read\n\nby Someone\nJune 20, 2025\n\n# Title") == "2025-06-20"


def test_rows_only_extracted_from_stamped_or_fee_tables():
    # an amortization table (no effective date, no "fee") must not become rate facts
    md = """
Balance over time

| Category | Interest | Principal | Remaining |
| --- | --- | --- | --- |
| 1 | $0 | $0 | $25,000 |
"""
    rows = rows_from_page("u", "t", md, PAGE_TYPE_CALCULATOR, "s", "2026-09-28", "2026-09-28")
    assert rows == []


def test_zero_percent_loan_rate_dropped_but_apy_kept():
    loan = """
Mortgage

| Program | Rate | Points | APR |
| --- | --- | --- | --- |
| 30-Year Fixed | 0.00% | 0 | 7.29% APR |

Rates are effective as of 09/25/2026.
"""
    rows = rows_from_page("u", "t", loan, PAGE_TYPE_RATES_HUB, "s", "2026-09-28", "2026-09-28")
    values = {(r.metric, r.value_low) for r in rows}
    assert ("apr", 7.29) in values
    assert ("rate", 0.0) not in values  # 0% loan rate is the Points column, dropped
