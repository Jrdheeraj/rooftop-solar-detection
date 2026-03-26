FROM python:3.11-slim

WORKDIR /app

# System deps for scientific / CV stack
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
# Install CPU-only PyTorch and torchvision
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu
# Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Make src importable as a package
ENV PYTHONPATH=/app/src

# Ensure outputs folders exist (defensive)
RUN mkdir -p outputs/overlays outputs/logs data/processed/google_images_all

# Expose the API port
EXPOSE 8002

# Run the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8002"]