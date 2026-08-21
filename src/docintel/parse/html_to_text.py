"""SEC filing HTML -> clean, readable text.

Filings are hostile HTML: inline styles everywhere, tables used both for layout
and for data, page-break artifacts, and XBRL wrappers. Rules applied here:

- script/style/head and XBRL ix:header metadata are dropped entirely
- tables are NOT dropped: each row is rendered as cells joined with " | ",
  because financial statements and risk tables carry real answer content
- non-breaking spaces and Unicode dashes are normalized; whitespace collapsed
- page artifacts (bare page numbers, horizontal-rule cruft) are stripped
"""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser, Node

_DROP_TAGS = {"script", "style", "head"}
# XBRL machine-readable metadata block; namespaced tags can't be CSS-selected
# by the modest engine, so it is removed via traversal.
_DROP_EXOTIC_TAGS = {"ix:header"}
_BLOCK_TAGS = {
    "p", "div", "br", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "li", "hr",
}
_PAGE_NUMBER_RE = re.compile(r"^\s*-?\s*\d{1,3}\s*-?\s*$")  # lone page numbers
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_SPACES_RE = re.compile(r"[ \t\f\v]+")


def _render_table(table: Node) -> str:
    rows: list[str] = []
    for tr in table.css("tr"):
        cells = [c.text(separator=" ", strip=True) for c in tr.css("td, th")]
        cells = [_SPACES_RE.sub(" ", c) for c in cells if c.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    exotic = [n for n in tree.root.traverse() if n.tag in _DROP_EXOTIC_TAGS]
    for node in exotic:
        node.decompose()

    # Render tables as text before the generic walk so cell adjacency survives.
    # Innermost-first: filings nest tables (layout around data), and replacing a
    # nested table with its text first means the outer table's cells see plain
    # text instead of duplicate <tr> rows.
    def _depth(node: Node) -> int:
        d = 0
        parent = node.parent
        while parent is not None:
            d += 1
            parent = parent.parent
        return d

    for table in sorted(tree.css("table"), key=_depth, reverse=True):
        table.replace_with(_render_table(table) + "\n")

    body = tree.body or tree.root
    text = body.text(separator="\n", strip=False) if body else ""

    # normalize characters
    text = text.replace("\xa0", " ").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "—")

    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _SPACES_RE.sub(" ", raw_line).strip()
        if _PAGE_NUMBER_RE.match(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()
