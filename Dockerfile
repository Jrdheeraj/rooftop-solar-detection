FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (required for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Filter out torch/torchvision from requirements to prevent overwriting CPU version with massive CUDA version
RUN grep -v "torch" requirements.txt > req_filtered.txt && \
    pip install --no-cache-dir -r req_filtered.txt

# Copy full project
COPY . .

# Make src importable
ENV PYTHONPATH=/app

# Create required folders (safe)
RUN mkdir -p outputs/overlays outputs/logs data/processed/google_images_all

# Start FastAPI using dynamic port (RENDER REQUIREMENT)
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]