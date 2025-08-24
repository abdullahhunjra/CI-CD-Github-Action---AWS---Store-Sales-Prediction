FROM python:3.10-slim

WORKDIR /app

# Install system-level build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel for smooth installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for Docker caching
COPY requirements.txt .

# Install dependencies with prebuilt wheels whenever possible
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
