FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level multimedia dependencies for OpenCV & audio handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    libgstreamer1.0-0 \
    alsa-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-compiled packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application layers
COPY . /app/

# Dedicated persistence volume directory for SQLite telemetry
RUN mkdir -p /app/data

CMD ["python", "main.py"]
