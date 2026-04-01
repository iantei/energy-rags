# ── Energy Policy RAG — Dockerfile ────────────────────────────────────────────
# Multi-stage build to keep the final image lean.
#
# This container runs the Python app (Gradio UI + ingestion engine).
# It uses system RAM for Polars ETL. GPU inference is handled by the
# separate Ollama container in docker-compose.yml.
#
# Build:  docker build -t energy-rag .
# Run:    docker compose up   (preferred — handles Ollama + networking)

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Skip Docling in Docker build — install separately if needed
# Docling auto-falls back to PyPDF when not installed
RUN pip install --upgrade pip && \
    grep -v "^docling" requirements.txt | pip install --no-cache-dir -r /dev/stdin

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY . .

RUN mkdir -p data/pdfs data/chroma_db && chmod -R 755 data/

EXPOSE 7860

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "app.py"]
