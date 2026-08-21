-- Enable pgvector. Applied by `docintel db upgrade` (see src/docintel/db.py).
CREATE EXTENSION IF NOT EXISTS vector;
