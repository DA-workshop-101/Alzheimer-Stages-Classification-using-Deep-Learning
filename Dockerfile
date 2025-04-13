# Stage 1: Build
FROM python:3.9-slim AS builder
# Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/lists

# Install Python Dependencies
COPY req_prod.txt .
RUN pip install --user -r req_prod.txt

# Stage 2: Final
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    rm -rf /var/lib/lists

#Copy Only Necessary files from the builder stage
COPY --from=builder /root/.local /root/.local
COPY models/ models/
COPY src/ src/
COPY webapp/backend/ webapp/backend/
COPY params.yaml .

#Set Environemnt Variables
ENV PATH=/root/.local/bin:$PATH

# Expose the port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn webapp.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
