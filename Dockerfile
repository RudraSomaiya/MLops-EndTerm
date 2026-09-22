FROM python:3.11-slim

# install uv for fast, cross-platform dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# install only production deps, no venv (Docker container is already isolated)
ENV UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --no-dev --frozen

COPY src/ src/
COPY app.py config.yaml ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
