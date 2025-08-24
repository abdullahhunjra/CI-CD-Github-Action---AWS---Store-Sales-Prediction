FROM python:3.8-slim

WORKDIR /app

# Install pip + upgrade to recent
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app/

EXPOSE 8000

# Run FastAPI with single worker (avoid duplicating 10GB+ model in memory)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
