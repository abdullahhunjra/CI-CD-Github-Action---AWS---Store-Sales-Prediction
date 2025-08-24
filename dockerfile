# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel to avoid source builds
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements.txt first for dependency caching
COPY requirements.txt .

# Install Python dependencies using prebuilt wheels only
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app code
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
