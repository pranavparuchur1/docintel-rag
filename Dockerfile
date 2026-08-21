FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
RUN pip install --no-cache-dir --no-deps -e .

CMD ["docintel", "--help"]
