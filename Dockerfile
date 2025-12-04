# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p outputs logs models data/processed data/raw

# Verify model weights exist or create placeholder
RUN if [ ! -f models/solar_model_best.pt ]; then \
        echo "Warning: models/solar_model_best.pt not found. Please download from Colab."; \
    fi

# Expose port (optional, for API server)
EXPOSE 5000 8000

# Health check: verify model can be loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from ultralytics import YOLO; print('YOLO loaded successfully')" || exit 1

# Default command: run batch inference
CMD ["python", "-m", "src.export_rooftop_json"]

# Optional entry point for flexibility
# Uncomment to use:
# ENTRYPOINT ["python"]
# CMD ["-m", "src.export_rooftop_json"]

# Alternative commands (override at runtime):
# docker run ... python -m src.build_final_predictions_json
# docker run ... python src/inference.py
# docker run ... python src/api.py