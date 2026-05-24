FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --quiet

COPY . .

RUN mkdir -p /app/data

RUN uv sync --no-dev --frozen

# Add venv to PATH so Railway can locate uvicorn
ENV PATH="/app/.venv/bin:$PATH"

# Verify the app can be imported before shipping
RUN python -c "from src.api.app import app; print('✓ app import OK')"

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
