# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by numpy/scikit-learn
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY app /app/app
COPY templates /app/templates
COPY static /app/static

# Ensure media dir exists
RUN mkdir -p /app/media

EXPOSE 8000

# Default envs (overridable)
ENV IMG_PATH="data/sample_image.png" \
    VARIANCE="0.95" \
    OUTPUT_DIR="output" \
    JPEG_QUALITY="85"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]