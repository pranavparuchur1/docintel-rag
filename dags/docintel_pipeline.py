"""Airflow DAG: the docintel pipeline as a daily batch.

Deliberately thin: every task shells out to the same `docintel` CLI a human
runs, so orchestration adds scheduling and retries — never hidden behavior.
Idempotency lives in the pipeline itself (accession-number natural keys,
content-hash embedding), which is what makes a retry safe here.

Deploy: mount this repo into an Airflow image that has the project installed
(`pip install -r requirements.txt && pip install -e .`), or point the tasks at
`docker compose run --rm app docintel ...` instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# The project corpus: (CIK, company) pairs; scope discipline is deliberate.
CORPUS = [
    ("320193", "apple"),
    ("789019", "microsoft"),
    ("1045810", "nvidia"),
    ("1318605", "tesla"),
    ("19617", "jpmorgan"),
    ("21344", "coca_cola"),
]

default_args = {
    "owner": "docintel",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),  # generous: EDGAR rate limits are polite, not fast
}

with DAG(
    dag_id="docintel_pipeline",
    description="EDGAR ingest -> chunk -> embed (incremental) -> eval report",
    schedule="0 6 * * 1-5",  # weekday mornings; EDGAR posts filings on business days
    start_date=datetime(2026, 8, 1),
    catchup=False,
    max_active_runs=1,  # the pipeline is idempotent, not concurrent-safe per stage
    default_args=default_args,
    tags=["docintel", "rag"],
) as dag:
    migrate = BashOperator(task_id="db_upgrade", bash_command="docintel db upgrade")

    ingest_tasks = [
        BashOperator(
            task_id=f"ingest_{name}",
            # sequential on purpose: one polite EDGAR client, not six parallel ones
            bash_command=f"docintel ingest --cik {cik} --forms 10-K,10-Q --limit 2",
        )
        for cik, name in CORPUS
    ]

    chunk = BashOperator(task_id="chunk", bash_command="docintel chunk --strategy all")

    # content-hash incrementality makes this a no-op unless filings changed
    embed = BashOperator(task_id="embed", bash_command="docintel embed --strategy all")

    evaluate = BashOperator(
        task_id="eval",
        bash_command="docintel eval --mode hybrid,agent",
    )

    migrate >> ingest_tasks[0]
    for upstream, downstream in zip(ingest_tasks, ingest_tasks[1:], strict=False):
        upstream >> downstream
    ingest_tasks[-1] >> chunk >> embed >> evaluate
