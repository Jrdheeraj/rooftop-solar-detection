FROM python:3.11-slim

WORKDIR /app

# System deps
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

# Install PyTorch (CPU)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Correct PYTHONPATH
ENV PYTHONPATH=/app

# Create folders
RUN mkdir -p outputs/overlays outputs/logs data/processed/google_images_all

# Use dynamic port for Render
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port $PORT"]