FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --quiet

COPY . .

RUN mkdir -p /app/data

RUN uv sync --no-dev --frozen

# Verify the app can be imported before shipping
RUN /app/.venv/bin/python -c "from src.api.app import app; print('✓ app import OK')"

CMD ["/bin/sh", "-c", "/app/.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
