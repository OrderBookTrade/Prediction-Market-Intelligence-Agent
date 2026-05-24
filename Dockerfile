FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --quiet

COPY . .

RUN uv sync --no-dev --frozen

CMD ["sh", "-c", "/app/.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
