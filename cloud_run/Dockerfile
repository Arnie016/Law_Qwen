FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port (Cloud Run uses PORT env var)
ENV PORT=8080
EXPOSE 8080

# Run
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}


