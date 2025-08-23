# ===========================
# Stage 1: Build dependencies
# ===========================
FROM python:3.13-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# Stage 2: Final image
# ===========================
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY src/ ./src
COPY data/ ./data
COPY README.md .
COPY params.yaml .
COPY enterprise_income_model_2025-08-21_14-26.joblib .  

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port if needed (for Streamlit)
EXPOSE 8501

# Set default command
CMD ["streamlit", "run", "src/Income_Prediction.py"]
