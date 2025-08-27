# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/app
COPY templates /app/templates
COPY static /app/static

EXPOSE 8001
ENV OLLAMA_HOST="http://ollama:11434"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8001"]

