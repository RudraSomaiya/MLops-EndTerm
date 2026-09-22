FROM python:3.11-slim

# install uv for fast, cross-platform dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# install only production deps (no dev extras), system python, no venv inside container
RUN uv sync --no-dev --system --frozen

COPY src/ src/
COPY app.py config.yaml ./

EXPOSE 8000

CMD ["uv", "run", "--system", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
