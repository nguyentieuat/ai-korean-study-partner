# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from api_evaluateASRneedleman import router as evaluate_needleman
# from api_evaluateMFAW2V import router as evaluate_mfa_w2v
from api_evaluateW2v2Forced import router as evaluate_w2v_forced
from api_pronunciationDataset import router as pronunciation_dataset_router


app = FastAPI(title="MFA/Needleman Evaluator")

# CORS (mở; chỉnh lại origin khi lên prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Routers
# app.include_router(evaluate_needleman)
# app.include_router(evaluate_mfa_w2v)
app.include_router(evaluate_w2v_forced)
app.include_router(pronunciation_dataset_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    # Chạy mfa service ở cổng 5002
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=True)
