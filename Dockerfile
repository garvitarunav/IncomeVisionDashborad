# ===========================
# Stage 1: Build dependencies
# ===========================
FROM python:3.10-slim AS builder   

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies into /install (not system-wide yet)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ===========================
# Stage 2: Final runtime image
# ===========================
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy only required project files
COPY src/ /app/src/
COPY data/ /app/data/
COPY params.yaml /app/
COPY enterprise_income_m* /app/
COPY dvc.yaml /app/
COPY README.md /app/

# Set Streamlit config
ENV PORT=8080
EXPOSE 8080

# Run Streamlit app (Railway sets $PORT dynamically)
CMD ["sh", "-c", "streamlit run src/Income_Prediction.py --server.port=$PORT --server.address=0.0.0.0"]
