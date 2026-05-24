FROM python:3.11-slim

WORKDIR /app

# Install uv — pip is available in python:3.11-slim
RUN pip install uv --quiet

# Copy everything first so hatchling can find README.md during package build
COPY . .

# Install production deps (builds local package via hatchling)
RUN uv sync --no-dev --frozen

# Use uv run so .venv is automatically on PATH
CMD ["sh", "-c", "uv run uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
