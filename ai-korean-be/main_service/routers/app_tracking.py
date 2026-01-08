# app_tracking.py
import uuid, time
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any

from fastapi import APIRouter, Query, Request, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (create_engine, Column, String, Integer, Float, Boolean,
                        DateTime, JSON, UniqueConstraint)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, case, and_

DB_URL = "sqlite:///tracking.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class EventORM(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    event_id = Column(String, nullable=False)     # UUID từ client
    user_id_hash = Column(String, nullable=False) # ẩn danh (hash)
    session_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)   # chat_turn / pronun_scored / topik_answered
    item_id = Column(String, nullable=True)       # PRON-xxx / Q1-0001 ...
    duration_ms = Column(Integer, nullable=True)  # latency đo bên FE
    score_overall = Column(Float, nullable=True)  # pronunciation
    cer = Column(Float, nullable=True)            # pronunciation
    is_correct = Column(Boolean, nullable=True)   # topik
    meta = Column(JSON, nullable=True)            # linh hoạt
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("event_id", name="uq_event_id"),)

Base.metadata.create_all(engine)

# ---- Pydantic ----
EventType = Literal["chat_turn", "pronun_scored", "topik_answered"]

class EventIn(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id_hash: str
    session_id: str
    event_type: EventType
    item_id: Optional[str] = None
    duration_ms: Optional[int] = None
    score_overall: Optional[float] = None
    cer: Optional[float] = None
    is_correct: Optional[bool] = None
    meta: Optional[Dict[str, Any]] = None

class BatchIn(BaseModel):
    events: List[EventIn]

router = APIRouter(tags=["tracking"])

@router.post("/api/track/event")
def track_event(ev: EventIn):
    db = SessionLocal()
    try:
        row = EventORM(**ev.dict())
        db.add(row)
        db.commit()
        return {"ok": True, "dedup": False}
    except IntegrityError:
        db.rollback()
        return {"ok": True, "dedup": True}  # sự kiện trùng -> bỏ qua
    finally:
        db.close()

@router.post("/api/track/batch")
def track_batch(batch: BatchIn):
    db = SessionLocal()
    created, dup = 0, 0
    try:
        for ev in batch.events:
            try:
                db.add(EventORM(**ev.dict()))
                db.commit()
                created += 1
            except IntegrityError:
                db.rollback()
                dup += 1
        return {"ok": True, "created": created, "duplicates": dup}
    finally:
        db.close()

@router.get("/api/track/summary")
def summary(session_id: Optional[str] = Query(None)):
    """
    Nếu có session_id: trả về thống kê cho đúng session đó.
    Nếu không có: trả về thống kê toàn cục (tất cả session).
    """
    db = SessionLocal()
    try:
        # ----- bộ lọc chung -----
        filters = []
        if session_id:
            filters.append(EventORM.session_id == session_id)

        # ----- CHAT -----
        chat_count, chat_latency_max = db.query(
            func.count(EventORM.id),
            func.coalesce(func.max(EventORM.duration_ms), 0),
        ).filter(
            and_(EventORM.event_type == "chat_turn", *filters)
        ).one()

        # ----- PRONUN -----
        pron_count, pron_latency_max, pron_score_avg, pron_cer_avg = db.query(
            func.count(EventORM.id),
            func.coalesce(func.max(EventORM.duration_ms), 0),
            func.coalesce(func.avg(EventORM.score_overall), 0.0),
            func.coalesce(func.avg(EventORM.cer), 0.0),
        ).filter(
            and_(EventORM.event_type == "pronun_scored", *filters)
        ).one()

        # ----- TOPIK -----
        topk_count, topk_latency_max, topk_correct = db.query(
            func.count(EventORM.id),
            func.coalesce(func.max(EventORM.duration_ms), 0),
            func.coalesce(func.sum(case((EventORM.is_correct == True, 1), else_=0)), 0),
        ).filter(
            and_(EventORM.event_type == "topik_answered", *filters)
        ).one()

        # ----- meta bổ sung -----
        if session_id:
            scope = "session"
            unique_sessions = 1
        else:
            scope = "global"
            unique_sessions = db.query(func.count(func.distinct(EventORM.session_id))).scalar() or 0

        out = {
            "scope": scope,
            "session_id": session_id,
            "unique_sessions": unique_sessions,

            "chat_turns": int(chat_count),
            "chat_latency_max_ms": int(chat_latency_max),

            "pronun_attempts": int(pron_count),
            "pronun_latency_max_ms": int(pron_latency_max),
            "pronun_score_avg": round(float(pron_score_avg), 4),
            "pronun_cer_avg": round(float(pron_cer_avg), 4),

            "topik_answered": int(topk_count),
            "topik_correct": int(topk_correct),
            "topik_accuracy": round((float(topk_correct) / topk_count), 4) if topk_count else 0.0,
            "topik_latency_max_ms": int(topk_latency_max),
        }
        return out
    finally:
        db.close()
