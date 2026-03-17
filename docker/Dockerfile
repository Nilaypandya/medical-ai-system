# ─────────────────────────────────────────────────────────────
# Medical AI System – Production Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build:
#   stage 1 (builder) : install Python deps into a venv
#   stage 2 (runtime) : copy venv + app code → lean final image
#
# Build:  docker build -t medical-ai:latest -f docker/Dockerfile .
# Run:    docker run -p 8000:8000 medical-ai:latest
# GPU:    docker run --gpus all -p 8000:8000 medical-ai:latest
# ─────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed to compile OpenCV + TF wheel dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ──────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Copy application source
COPY --chown=appuser:appuser app/      ./app/
COPY --chown=appuser:appuser model/    ./model/
COPY --chown=appuser:appuser training/ ./training/

USER appuser

# Expose API port
EXPOSE 8000

# Health-check: Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
# - workers=1 keeps a single model copy in memory (scale with multiple containers)
# - --host 0.0.0.0 is required inside Docker (default 127.0.0.1 won't be reachable)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
