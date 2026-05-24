FROM python:3.11-slim

WORKDIR /app

# Install uv (pip is available in python:3.11-slim)
RUN pip install uv --quiet

# Copy dependency manifest first (cache layer)
COPY pyproject.toml uv.lock ./

# Install production deps into .venv
RUN uv sync --no-dev --frozen

# Copy source code
COPY . .

# Expose dynamic port (Railway injects $PORT)
EXPOSE 8000

# Use uv run so the .venv is automatically activated
CMD ["sh", "-c", "uv run uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
