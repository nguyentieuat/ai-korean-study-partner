# app.py (FastAPI)
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import httpx

from api_tts import router as tts_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # HTTP client shared (keep-alive), không đặt timeout để phù hợp file dài
    app.state.http = httpx.AsyncClient(timeout=None, follow_redirects=True)
    yield
    await app.state.http.aclose()

app = FastAPI(title="Main Orchestrator", lifespan=lifespan)

# CORS mở (chỉnh lại khi lên prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Routers
app.include_router(tts_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == '__main__':
    # Chạy main service ở cổng 5004
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5004, reload=True)