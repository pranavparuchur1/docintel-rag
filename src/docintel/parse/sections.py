"""Detect 'Item X' section headings in cleaned filing text.

Documented heuristic (measured, not assumed — see `docintel parse-report`):

1. A candidate heading is a LINE starting with "PART <roman>" or
   "Item <n><letter?>", at most 150 chars.
2. Table-of-contents entries are excluded: tables render as pipe-joined rows
   here, so a TOC entry looks like "Item 1A. | Risk Factors | 5" — pipe cells
   with a bare page number in the last cell. A pipe row WITHOUT a trailing
   number is kept (some filings style real headings inside layout tables).
3. Item numbers repeat across parts in a 10-Q, so keys are part-qualified for
   10-Q ("I.2", "II.1A") and bare for 10-K ("1A", "7").
4. For duplicate keys, a TITLED occurrence ("Item 1A. Risk Factors") always
   beats a bare one ("Item 1A"): several filers (e.g. Microsoft) print bare
   "Item 1A" running page headers on every page, so bare lines are only a
   fallback. Among equals the LAST occurrence wins, because the TOC precedes
   the body and inline cross-references don't start a line.

Known failure modes: filings that render headings as images, exotic layouts
where the heading shares a line with body text, and cover-page item mentions.
The hit rate against expected key sections is reported per corpus rather than
assumed to be 100%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PART_RE = re.compile(r"^part\s+([ivxl]+)\b", re.IGNORECASE)
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)[.:]?\s*(.*)$", re.IGNORECASE)
_PAGE_CELL_RE = re.compile(r"^\d{1,3}$")
MAX_HEADING_LEN = 150

# The sections a retrieval system must be able to cite for this project's
# question types; the hit rate is measured against these.
EXPECTED_ITEMS = {
    "10-K": {"1", "1A", "7", "7A", "8"},
    "10-Q": {"I.1", "I.2", "II.1A"},
}


@dataclass(frozen=True)
class Section:
    key: str  # "1A" (10-K) or "II.1A" (10-Q, part-qualified)
    title: str
    start: int  # char offset of heading line start in the cleaned text
    end: int  # char offset where the section ends (next heading or EOF)

    @property
    def label(self) -> str:
        title = f" {self.title}" if self.title else ""
        return f"Item {self.key.split('.')[-1]}.{title}"


def _is_toc_row(line: str) -> bool:
    if "|" not in line:
        return False
    last_cell = line.rsplit("|", 1)[-1].strip()
    return bool(_PAGE_CELL_RE.match(last_cell))


def detect_sections(text: str, form_type: str) -> list[Section]:
    part_qualified = form_type.upper().startswith("10-Q")
    current_part = "I"
    # key -> (start, title, titled): titled occurrences beat bare page headers;
    # among equals the last occurrence wins.
    found: dict[str, tuple[int, str, bool]] = {}

    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        length_ok = 0 < len(stripped) <= MAX_HEADING_LEN
        if length_ok and not _is_toc_row(stripped):
            part_match = PART_RE.match(stripped)
            if part_match:
                current_part = part_match.group(1).upper()
            else:
                item_match = ITEM_RE.match(stripped.replace(" | ", " "))
                if item_match:
                    number = item_match.group(1).upper()
                    title = item_match.group(2).strip(" .|-—")
                    key = f"{current_part}.{number}" if part_qualified else number
                    titled = bool(title)
                    previous = found.get(key)
                    if previous is None or titled or not previous[2]:
                        found[key] = (offset, title, titled)
        offset += len(line) + 1  # +1 for the newline split() removed

    ordered = sorted(
        ((key, start, title) for key, (start, title, _titled) in found.items()),
        key=lambda row: row[1],
    )
    sections: list[Section] = []
    for i, (key, start, title) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
        sections.append(Section(key=key, title=title, start=start, end=end))
    return sections


def section_hit_rate(sections: list[Section], form_type: str) -> tuple[set[str], set[str]]:
    """Return (expected keys, found-of-expected keys) for one document."""
    expected = EXPECTED_ITEMS.get(form_type.upper(), set())
    found = {s.key for s in sections} & expected
    return expected, found
