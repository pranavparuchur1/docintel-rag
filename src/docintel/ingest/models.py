from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Filing:
    """One filing as described by EDGAR's submissions index."""

    accession_no: str  # e.g. "0000320193-24-000123"
    cik: str  # normalized, no zero-padding
    company: str
    form_type: str
    filing_date: date
    period: date | None
    primary_document: str  # e.g. "aapl-20240928.htm"

    @property
    def accession_nodash(self) -> str:
        return self.accession_no.replace("-", "")

    @property
    def source_url(self) -> str:
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_nodash}/{self.primary_document}"
        )
