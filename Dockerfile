# Dockerfile for rooftop-solar-detection

FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and important folders
COPY src/ src/
COPY data/raw/ data/raw/
COPY models/ models/
COPY outputs/ outputs/

# Default command: print help
CMD ["python", "src/build_final_predictions_json.py"]