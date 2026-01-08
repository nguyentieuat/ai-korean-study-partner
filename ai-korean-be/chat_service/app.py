# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.api_conversation import router as conversation_router

app = FastAPI(title="Chat Service")

# CORS (tùy chỉnh origins khi deploy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Đăng ký router
app.include_router(conversation_router)

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    # Chạy chat service ở cổng 5001
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)