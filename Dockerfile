FROM python:3.10-slim

# Build args for proxy (pass via docker compose build --build-arg)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY="localhost,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FUNASR_MODEL=sensevoice \
    FUNASR_DEVICE=cpu \
    FUNASR_PORT=8000

WORKDIR /app

# Set proxy for apt in this build step
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/99proxy; \
        echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/99proxy; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
       ffmpeg \
       git \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install funasr fastapi "uvicorn[standard]" python-multipart

COPY server.py /app/server.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('FUNASR_PORT', '8000'), timeout=3).read()" || exit 1

CMD ["sh", "-c", "python server.py --host 0.0.0.0 --port ${FUNASR_PORT:-8000} --device ${FUNASR_DEVICE:-cpu} --model ${FUNASR_MODEL:-sensevoice}"]
