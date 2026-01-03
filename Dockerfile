FROM python:3-slim

WORKDIR /app

# 1. Install system dependencies (needed for OpenCV/Matplotlib backends + build tools)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python packages
# - opencv-python-headless: for cv2
# - numpy: for array manipulation
# - matplotlib: for plotting (plt)
RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    matplotlib