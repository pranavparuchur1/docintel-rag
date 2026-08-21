from pathlib import Path

import pytest

from docintel.parse import parse_filing
from docintel.parse.html_to_text import extract_text
from docintel.parse.sections import detect_sections, section_hit_rate

FIXTURE = Path(__file__).parent / "fixtures" / "mini10k.html"


@pytest.fixture(scope="module")
def mini_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def mini_text(mini_html) -> str:
    return extract_text(mini_html)


def test_script_and_style_are_dropped(mini_text):
    assert "tracker" not in mini_text
    assert "font-size" not in mini_text


def test_tables_render_as_readable_rows(mini_text):
    assert "Net revenue | $1,204 | $1,075" in mini_text


def test_nested_tables_do_not_duplicate_rows():
    html = (
        "<table><tr><td>outer</td><td>"
        "<table><tr><td>inner-a</td><td>inner-b</td></tr></table>"
        "</td></tr></table>"
    )
    text = extract_text(html)
    assert text.count("inner-a") == 1


def test_page_number_artifacts_removed(mini_text):
    assert "- 12 -" not in mini_text


def test_sections_detected_and_toc_excluded(mini_text):
    sections = detect_sections(mini_text, "10-K")
    keys = [s.key for s in sections]
    assert keys == ["1", "1A", "7", "7A", "8"]
    # the real Item 1A heading is in the body, past the TOC table
    item_1a = next(s for s in sections if s.key == "1A")
    assert "single contract manufacturer" in mini_text[item_1a.start : item_1a.end]
    assert "Net revenue" not in mini_text[item_1a.start : item_1a.end]  # that's Item 7


def test_hit_rate_is_full_on_fixture(mini_text):
    expected, found = section_hit_rate(detect_sections(mini_text, "10-K"), "10-K")
    assert found == expected == {"1", "1A", "7", "7A", "8"}


def test_10q_keys_are_part_qualified():
    text = (
        "PART I — FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements\n" + ("balance sheet line\n" * 5) +
        "Item 2. Management's Discussion and Analysis\n" + ("mdna line\n" * 5) +
        "PART II — OTHER INFORMATION\n"
        "Item 1. Legal Proceedings\n" + ("legal line\n" * 5) +
        "Item 1A. Risk Factors\n" + ("risk line\n" * 5)
    )
    keys = [s.key for s in detect_sections(text, "10-Q")]
    assert keys == ["I.1", "I.2", "II.1", "II.1A"]


def test_parse_filing_section_at(mini_html):
    parsed = parse_filing(mini_html, "10-K")
    item_7 = next(s for s in parsed.sections if s.key == "7")
    assert parsed.section_at(item_7.start + 10).key == "7"
    assert parsed.section_at(0) is None  # cover page precedes any section
