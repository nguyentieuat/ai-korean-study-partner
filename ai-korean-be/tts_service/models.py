from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    text: str = Field(..., description="Văn bản tiếng Hàn cần đọc")
    slow_factor: float = Field(0.9, ge=0.5, le=1.5, description="<1 = chậm hơn (giữ cao độ)")
    pause_ms: int = Field(400, ge=0, le=5000, description="khoảng lặng giữa câu (ms)")
    cache: bool = True
    filename: str | None = Field(None, description="Tên file .wav khi tải về")

class TTSResponse(BaseModel):
    sampling_rate: int
    cache_key: str
    bytes: int | None = None
