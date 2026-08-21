# Pipeline commands run inside the app container (Python 3.11, same as CI),
# so results are identical for anyone who clones the repo. Test/lint run in
# whatever environment invokes make (local venv or CI).

COMPOSE_RUN = docker compose run --rm app

.PHONY: up down build ingest embed query eval test lint db-upgrade db-check

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
