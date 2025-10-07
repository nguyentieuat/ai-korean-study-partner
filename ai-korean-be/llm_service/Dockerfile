FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps: có curl cho healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# module FastAPI của bạn (đổi nếu file khác 'server.py' hoặc 'main.py')
# ví dụ: "server:app" hoặc "main:app"
ENV APP_MODULE="server:app"
ENV PORT=5006

EXPOSE 5006

# chạy uvicorn, dùng APP_MODULE và PORT từ env
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5006"]
