"""Filing selection is pure and unit-tested; idempotency is proven against Postgres."""

import hashlib

from docintel.ingest.service import ingest_documents, normalize_cik, select_filings

SUBMISSIONS = {
    "cik": 320193,
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0000320193-24-000123",
                "0000320193-24-000081",
                "0000320193-24-000069",
                "0000320193-23-000106",
                "0000320193-23-000077",
            ],
            "form": ["10-K", "10-Q", "8-K", "10-K", "10-Q"],
            "filingDate": ["2024-11-01", "2024-08-02", "2024-05-03", "2023-11-03", "2023-08-04"],
            "reportDate": ["2024-09-28", "2024-06-29", "", "2023-09-30", "2023-07-01"],
            "primaryDocument": [
                "aapl-20240928.htm",
                "aapl-20240629.htm",
                "aapl-8k.htm",
                "aapl-20230930.htm",
                "aapl-20230701.htm",
            ],
        }
    },
}

FAKE_DOC = b"<html><body>Item 1A. Risk Factors ...</body></html>"


class FakeEdgarClient:
    def __init__(self):
        self.downloads = 0

    def get_json(self, url):
        assert url == "https://data.sec.gov/submissions/CIK0000320193.json"
        return SUBMISSIONS

    def download(self, url, dest):
        self.downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(FAKE_DOC)
        return hashlib.sha256(FAKE_DOC).hexdigest(), len(FAKE_DOC)


def test_normalize_cik_strips_padding():
    assert normalize_cik("0000320193") == "320193"


def test_select_filings_filters_sorts_and_limits():
    filings = select_filings(SUBMISSIONS, forms=["10-K", "10-Q"], limit=1)
    assert {f.form_type for f in filings} == {"10-K", "10-Q"}
    by_form = {f.form_type: f for f in filings}
    assert by_form["10-K"].accession_no == "0000320193-24-000123"  # newest 10-K
    assert by_form["10-Q"].accession_no == "0000320193-24-000081"
    assert all(f.form_type != "8-K" for f in filings)


def test_select_filings_builds_archive_url():
    filing = select_filings(SUBMISSIONS, forms=["10-K"], limit=1)[0]
    assert filing.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )


def test_ingest_twice_is_idempotent(conn, settings):
    client = FakeEdgarClient()

    first = ingest_documents(conn, client, settings, cik="320193", forms=["10-K", "10-Q"], limit=2)
    assert first.downloaded == 4
    assert client.downloads == 4

    second = ingest_documents(conn, client, settings, cik="320193", forms=["10-K", "10-Q"], limit=2)
    assert second.downloaded == 0
    assert second.skipped == 4
    assert client.downloads == 4  # no new network calls for documents

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone()[0] == 4


def test_missing_raw_file_triggers_redownload(conn, settings):
    client = FakeEdgarClient()
    ingest_documents(conn, client, settings, cik="320193", forms=["10-K"], limit=1)

    with conn.cursor() as cur:
        cur.execute("SELECT raw_path FROM documents")
        raw_path = cur.fetchone()[0]
    (settings.data_dir / raw_path).unlink()

    stats = ingest_documents(conn, client, settings, cik="320193", forms=["10-K"], limit=1)
    assert stats.refreshed == 1
    assert (settings.data_dir / raw_path).exists()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        assert cur.fetchone()[0] == 1  # still one row — upsert, not insert
