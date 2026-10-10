# Multi-stage / lightweight production container for CloudWise on Google Cloud Run
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install security updates and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-privileged user for security compliance (Principle of Least Privilege)
RUN useradd -m -u 1001 appuser

# Copy application source code
COPY app.py .
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/
RUN mkdir -p ./reports && chown -R appuser:appuser /app

USER appuser

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Health check to ensure Streamlit is serving
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Start Streamlit listening on Cloud Run PORT environment variable
CMD exec streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false
