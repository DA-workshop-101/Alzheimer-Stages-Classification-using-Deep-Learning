# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set environment to install Python packages in a custom path
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUSERBASE=/install/deps

# Copy only the requirements file
COPY req_prod.txt .

# Install Python dependencies to custom path
RUN pip install --user -r req_prod.txt

# ---------------------------------------------------------

# Stage 2: Final image
FROM python:3.9-slim

WORKDIR /app

# System deps for runtime (OpenCV etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install/deps /root/.local

# Update PATH
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local

# Copy application code
COPY models/ models/
COPY src/ src/
COPY webapp/backend/ webapp/backend/
COPY params.yaml .

# Expose the app port
EXPOSE 8000

# Default command
CMD ["sh", "-c", "uvicorn webapp.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
