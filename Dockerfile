# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY scripts/ ./scripts/
COPY app.py .

# Create tmp directory and copy vector_db into it
RUN mkdir -p /tmp/vector_db
COPY vector_db/ /tmp/vector_db/

# Install FastAPI and uvicorn
RUN pip install fastapi uvicorn

# Set environment variable for vector db path
ENV VECTOR_DB_PATH=/tmp/vector_db

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]