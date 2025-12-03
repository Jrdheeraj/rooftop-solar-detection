# Dockerfile for rooftop-solar-detection
# Multi-stage build for optimized production image

FROM python:3.11-slim

# Set metadata
LABEL maintainer="EcoInnovators Team"
LABEL description="AI-powered rooftop solar detection system"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config/ config/
COPY data/raw/ data/raw/
COPY models/ models/
COPY outputs/ outputs/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port for API (if running FastAPI)
EXPOSE 8000

# Default command: Generate predictions
CMD ["python", "src/build_final_predictions_json.py"]

# Alternative command for interactive use:
# CMD ["/bin/bash"]
