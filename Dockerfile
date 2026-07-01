FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy into subdirectory so from smallest_test.xxx imports work
WORKDIR /app
COPY requirements.txt smallest_test/
RUN pip install --no-cache-dir -r smallest_test/requirements.txt

COPY . smallest_test/
WORKDIR /app/smallest_test

# Coolify sets PORT env var, default to 8002
ENV PORT=8002

EXPOSE ${PORT}

CMD ["python", "standalone_server.py"]
