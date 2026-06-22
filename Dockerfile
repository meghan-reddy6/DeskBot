# ==========================================
# STAGE 1: Builder / Dependency Compilation
# ==========================================
FROM python:3.12-slim AS builder

# Optimize Python runtime for containerization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies required for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Establish an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Upgrade pip and install core dependencies (onnxruntime, opencv-python, etc.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Minimal Runtime Environment
# ==========================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install critical runtime libraries for OpenCV and GStreamer (Qualcomm/Rubik Pi support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the core python application scripts
COPY *.py ./

# Copy the pre-exported ONNX edge model
# (Ensure yolov8n-pose.onnx is present in the build context directory)
COPY yolov8n-pose.onnx ./

# Create data directory for named volume persistence mapping
RUN mkdir -p /app/data

# Launch the Daemon Entrypoint
CMD ["python", "main.py"]
