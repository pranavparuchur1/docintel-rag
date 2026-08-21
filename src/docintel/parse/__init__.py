"""docintel.parse — filing HTML to clean text plus detected sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docintel.parse.html_to_text import extract_text
from docintel.parse.sections import Section, detect_sections


@dataclass(frozen=True)
class ParsedDoc:
    text: str
    sections: list[Section]

    def section_at(self, char_offset: int) -> Section | None:
        for section in self.sections:
            if section.start <= char_offset < section.end:
                return section
        return None


def load_raw_text(path: Path) -> str:
    """EDGAR serves mostly UTF-8, but older filings are windows-1252."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse_filing(html: str, form_type: str) -> ParsedDoc:
    text = extract_text(html)
    return ParsedDoc(text=text, sections=detect_sections(text, form_type))
