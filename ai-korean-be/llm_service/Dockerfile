# Dockerfile
FROM python:3.11-slim

# --- System setup (Pillow & performance) ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Pillow thường cần libjpeg/zlib để xử lý ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install Python deps ---
# Giả sử bạn build ở thư mục llm-service (chứa requirements.txt, app.py, vision_adapter.py, ...)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# --- Copy source code ---
COPY . /app

# --- Default envs (có thể override khi chạy) ---
# Nếu Ollama chạy ngoài container:
# - Trên Windows/macOS Docker Desktop: dùng http://host.docker.internal:11434
# - Trên Linux: dùng IP host thật (vd 172.17.0.1) hoặc chạy chung mạng với container Ollama
ENV OLLAMA_HOST="http://host.docker.internal:11434" \
    DEFAULT_MODEL="qwen2.5:7b-instruct" \
    HTTP_TIMEOUT="360" \
    VISION_MODEL="moondream" \
    PORT="5006"

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s CMD curl -fsS http://127.0.0.1:5006/health || exit 1
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5006"]
