FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv --quiet

# Copy source
COPY . .

# Create data directory (gitignored, needed for SQLite)
RUN mkdir -p /app/data

# Install production deps
RUN uv sync --no-dev --frozen

CMD ["/bin/sh", "-c", "/app/.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
