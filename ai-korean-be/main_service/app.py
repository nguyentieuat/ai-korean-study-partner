# app.py (FastAPI)
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import httpx

from api_conversation import router as conversation_router
from api_pronunciation import router as pronunciation_router
from api_topikgenerate import router as topik_router
from api_cooperate import router as cooperate_router

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

# Static /static/*
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(conversation_router)
app.include_router(pronunciation_router)
app.include_router(topik_router)
app.include_router(cooperate_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == '__main__':
    # Chạy main service ở cổng 5000
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)