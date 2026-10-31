# ==============================================================================
# OmniDoc-Vision Production & Development Container Image
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install minimal system dependencies for PDF/image rendering and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy project source and configuration
COPY pyproject.toml README.md main.py app.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
COPY configs/ ./configs/
COPY public/ ./public/
COPY .chainlit/ ./.chainlit/
COPY data/ ./data/

# Create non-root runtime user and ensure writable data/eval directories
RUN useradd --create-home --shell /bin/bash omnidoc && \
    mkdir -p /app/data/raw_pdfs /app/data/crops /app/data/bm25_index /app/evals/results && \
    chown -R omnidoc:omnidoc /app

USER omnidoc

EXPOSE 8001 8002

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
