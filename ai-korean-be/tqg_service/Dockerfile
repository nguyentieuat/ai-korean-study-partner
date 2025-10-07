FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Gợi ý: thay openai-whisper bằng faster-whisper để nhanh hơn trên CPU:
# sed -i 's/openai-whisper/faster-whisper/g' requirements.txt  # nếu muốn
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m app && chown -R app:app /app
USER app


ENV PORT=5003
ENV CHAT_SERVICE_URL=http://127.0.0.1:5001
ENV MFA_SERVICE_URL=http://127.0.0.1:5002
ENV TQG_SERVICE_URL=http://127.0.0.1:5003
ENV TTS_SERVICE_URL=http://127.0.0.1:5004
ENV ASR_SERVICE_URL=http://127.0.0.1:5005
ENV LLM_SERVICE_URL=http://127.0.0.1:5006

ENV LLM_LOCAL_ENABLED=true
ENV LLM_LOCAL_MODEL=qwen2.5:7b-instruct

ENV DIALOGUE_MODEL=gpt-4o-mini
ENV GRAMMAR_MODEL=gpt-4o-mini
ENV OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

EXPOSE 5003
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s CMD curl -fsS http://127.0.0.1:5003/health || exit 1
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","5003"]
