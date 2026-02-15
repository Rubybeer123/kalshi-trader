# Production Dockerfile for Kalshi Trader
# Multi-stage build for optimized image size

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Security: Create non-root user early
RUN groupadd -r trader && useradd -r -g trader trader

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Security: Create non-root user
RUN groupadd -r trader && useradd -r -g trader -d /app trader

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libta-lib0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=trader:trader src/ ./src/
COPY --chown=trader:trader config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/secrets && \
    chown -R trader:trader /app

# Switch to non-root user
USER trader

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_ENV=production \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose metrics port
EXPOSE 8080

# Run the trading bot
CMD ["python", "-m", "src.main"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional)
# -----------------------------------------------------------------------------
FROM production AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install test dependencies
COPY requirements.txt .
RUN pip install pytest pytest-asyncio pytest-cov black ruff mypy

USER trader

# Override for development
ENV APP_ENV=development \
    LOG_FORMAT=text

CMD ["python", "-m", "pytest", "tests/", "-v"]
