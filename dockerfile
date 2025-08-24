FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY app/ ./app/

EXPOSE 8000

# Run with single worker (avoid multiple copies of 10GB model in RAM)
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
