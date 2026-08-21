# Pipeline commands run inside the app container (Python 3.11, same as CI),
# so results are identical for anyone who clones the repo. Test/lint run in
# whatever environment invokes make (local venv or CI).

COMPOSE_RUN = docker compose run --rm app

.PHONY: up down build ingest corpus embed query eval test lint db-upgrade db-check

up:
	docker compose up -d db

down:
	docker compose down

build:
	docker compose build app

db-upgrade:
	$(COMPOSE_RUN) docintel db upgrade

db-check:
	$(COMPOSE_RUN) docintel db check

# usage: make ingest CIK=0000320193 FORMS=10-K,10-Q LIMIT=3
ingest:
	$(COMPOSE_RUN) docintel ingest --cik $(CIK) --forms $(FORMS) --limit $(LIMIT)

# The project corpus: 6 companies x (2 10-Ks + 1 10-Q) = 18 documents.
# AAPL, MSFT, NVDA, TSLA, JPM, KO — enough overlap for cross-company and
# temporal eval questions without scaling the corpus instead of the pipeline.
CORPUS_CIKS = 320193 789019 1045810 1318605 19617 21344

corpus:
	for cik in $(CORPUS_CIKS); do \
		$(COMPOSE_RUN) docintel ingest --cik $$cik --forms 10-K --limit 2 && \
		$(COMPOSE_RUN) docintel ingest --cik $$cik --forms 10-Q --limit 1 || exit 1; \
	done

embed:
	$(COMPOSE_RUN) docintel embed

# usage: make query Q="What are Apple's main risk factors?"
query:
	$(COMPOSE_RUN) docintel query "$(Q)"

eval:
	$(COMPOSE_RUN) docintel eval

test:
	pytest

lint:
	ruff check .
