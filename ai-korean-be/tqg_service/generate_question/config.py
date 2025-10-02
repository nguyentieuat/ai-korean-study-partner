# generate_question/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional
from .utils import norm

# ====== CẤU HÌNH DỮ LIỆU (ENV → Path) ======
TOPIKI14_LISTEN_DIR   = Path(os.getenv("TOPIKI14_LISTEN_DIR",   "data/topik1/listen/1-4/OUT")).resolve()
TOPIKI56_LISTEN_DIR   = Path(os.getenv("TOPIKI56_LISTEN_DIR",   "data/topik1/listen/5-6/OUT")).resolve()
TOPIKI710_LISTEN_DIR  = Path(os.getenv("TOPIKI710_LISTEN_DIR",  "data/topik1/listen/7-10/OUT")).resolve()
TOPIKI1114_LISTEN_DIR = Path(os.getenv("TOPIKI1114_LISTEN_DIR", "data/topik1/listen/11-14/OUT")).resolve()
TOPIKI1516_LISTEN_DIR = Path(os.getenv("TOPIKI1516_LISTEN_DIR", "data/topik1/listen/15-16/OUT")).resolve()
TOPIKI1721_LISTEN_DIR = Path(os.getenv("TOPIKI1721_LISTEN_DIR", "data/topik1/listen/17-21/OUT")).resolve()
TOPIKI2224_LISTEN_DIR = Path(os.getenv("TOPIKI2224_LISTEN_DIR", "data/topik1/listen/22-24/OUT")).resolve()
TOPIKI2526_LISTEN_DIR = Path(os.getenv("TOPIKI2526_LISTEN_DIR", "data/topik1/listen/25-26/OUT")).resolve()
TOPIKI2728_LISTEN_DIR = Path(os.getenv("TOPIKI2728_LISTEN_DIR", "data/topik1/listen/27-28/OUT")).resolve()
TOPIKI2930_LISTEN_DIR = Path(os.getenv("TOPIKI2930_LISTEN_DIR", "data/topik1/listen/29-30/OUT")).resolve()

TOPIKI3133_READ_DIR   = Path(os.getenv("TOPIKI3133_READ_DIR",   "data/topik1/reading/31-33/OUT")).resolve()
TOPIKI3439_READ_DIR   = Path(os.getenv("TOPIKI3439_READ_DIR",   "data/topik1/reading/34-39/OUT")).resolve()
TOPIKI4042_READ_DIR   = Path(os.getenv("TOPIKI4042_READ_DIR",   "data/topik1/reading/40-42/OUT")).resolve()
TOPIKI4345_READ_DIR   = Path(os.getenv("TOPIKI4345_READ_DIR",   "data/topik1/reading/43-45/OUT")).resolve()
TOPIKI4648_READ_DIR   = Path(os.getenv("TOPIKI4648_READ_DIR",   "data/topik1/reading/46-48/OUT")).resolve()
TOPIKI4956_READ_DIR   = Path(os.getenv("TOPIKI4956_READ_DIR",   "data/topik1/reading/49-56/OUT")).resolve()
TOPIKI5758_READ_DIR   = Path(os.getenv("TOPIKI5758_READ_DIR",   "data/topik1/reading/57-58/OUT")).resolve()
TOPIKI5962_READ_DIR   = Path(os.getenv("TOPIKI5962_READ_DIR",   "data/topik1/reading/59-62/OUT")).resolve()
TOPIKI6364_READ_DIR   = Path(os.getenv("TOPIKI6364_READ_DIR",   "data/topik1/reading/63-64/OUT")).resolve()
TOPIKI6570_READ_DIR   = Path(os.getenv("TOPIKI6570_READ_DIR",   "data/topik1/reading/65-70/OUT")).resolve()

TOPIKII13_LISTEN_DIR  = Path(os.getenv("TOPIKII13_LISTEN_DIR",  "data/topik2/listen/1-3/OUT")).resolve()
TOPIKII48_LISTEN_DIR  = Path(os.getenv("TOPIKII48_LISTEN_DIR",  "data/topik2/listen/4-8/OUT")).resolve()
TOPIKII912_LISTEN_DIR = Path(os.getenv("TOPIKII912_LISTEN_DIR", "data/topik2/listen/9-12/OUT")).resolve()
TOPIKII1316_LISTEN_DIR= Path(os.getenv("TOPIKII1316_LISTEN_DIR","data/topik2/listen/13-16/OUT")).resolve()
TOPIKII1720_LISTEN_DIR= Path(os.getenv("TOPIKII1720_LISTEN_DIR","data/topik2/listen/17-20/OUT")).resolve()
TOPIKII2130_LISTEN_DIR= Path(os.getenv("TOPIKII2130_LISTEN_DIR","data/topik2/listen/21-30/OUT")).resolve()
TOPIKII3132_LISTEN_DIR= Path(os.getenv("TOPIKII3132_LISTEN_DIR","data/topik2/listen/31-32/OUT")).resolve()
TOPIKII3336_LISTEN_DIR= Path(os.getenv("TOPIKII3336_LISTEN_DIR","data/topik2/listen/33-36/OUT")).resolve()
TOPIKII3738_LISTEN_DIR= Path(os.getenv("TOPIKII3738_LISTEN_DIR","data/topik2/listen/37-38/OUT")).resolve()
TOPIKII3940_LISTEN_DIR= Path(os.getenv("TOPIKII3940_LISTEN_DIR","data/topik2/listen/39-40/OUT")).resolve()
TOPIKII4142_LISTEN_DIR= Path(os.getenv("TOPIKII4142_LISTEN_DIR","data/topik2/listen/41-42/OUT")).resolve()
TOPIKII4344_LISTEN_DIR= Path(os.getenv("TOPIKII4344_LISTEN_DIR","data/topik2/listen/43-44/OUT")).resolve()
TOPIKII4546_LISTEN_DIR= Path(os.getenv("TOPIKII4546_LISTEN_DIR","data/topik2/listen/45-46/OUT")).resolve()
TOPIKII4748_LISTEN_DIR= Path(os.getenv("TOPIKII4748_LISTEN_DIR","data/topik2/listen/47-48/OUT")).resolve()
TOPIKII4950_LISTEN_DIR= Path(os.getenv("TOPIKII4950_LISTEN_DIR","data/topik2/listen/49-50/OUT")).resolve()

TOPIKII12_READ_DIR    = Path(os.getenv("TOPIKII12_READ_DIR",    "data/topik2/reading/1-2/OUT")).resolve()
TOPIKII34_READ_DIR    = Path(os.getenv("TOPIKII34_READ_DIR",    "data/topik2/reading/3-4/OUT")).resolve()
TOPIKII58_READ_DIR    = Path(os.getenv("TOPIKII58_READ_DIR",    "data/topik2/reading/5-8/OUT")).resolve()
TOPIKII912_READ_DIR   = Path(os.getenv("TOPIKII912_READ_DIR",   "data/topik2/reading/9-12/OUT")).resolve()
TOPIKII1315_READ_DIR  = Path(os.getenv("TOPIKII1315_READ_DIR",  "data/topik2/reading/13-15/OUT")).resolve()
TOPIKII1618_READ_DIR  = Path(os.getenv("TOPIKII1618_READ_DIR",  "data/topik2/reading/16-18/OUT")).resolve()
TOPIKII1922_READ_DIR  = Path(os.getenv("TOPIKII1922_READ_DIR",  "data/topik2/reading/19-22/OUT")).resolve()
TOPIKII2324_READ_DIR  = Path(os.getenv("TOPIKII2324_READ_DIR",  "data/topik2/reading/23-24/OUT")).resolve()
TOPIKII2527_READ_DIR  = Path(os.getenv("TOPIKII2527_READ_DIR",  "data/topik2/reading/25-27/OUT")).resolve()
TOPIKII2831_READ_DIR  = Path(os.getenv("TOPIKII2831_READ_DIR",  "data/topik2/reading/28-31/OUT")).resolve()
TOPIKII3234_READ_DIR  = Path(os.getenv("TOPIKII3234_READ_DIR",  "data/topik2/reading/32-34/OUT")).resolve()
TOPIKII3538_READ_DIR  = Path(os.getenv("TOPIKII3538_READ_DIR",  "data/topik2/reading/35-38/OUT")).resolve()
TOPIKII3941_READ_DIR  = Path(os.getenv("TOPIKII3941_READ_DIR",  "data/topik2/reading/39-41/OUT")).resolve()
TOPIKII4243_READ_DIR  = Path(os.getenv("TOPIKII4243_READ_DIR",  "data/topik2/reading/42-43/OUT")).resolve()
TOPIKII4445_READ_DIR  = Path(os.getenv("TOPIKII4445_READ_DIR",  "data/topik2/reading/44-45/OUT")).resolve()
TOPIKII4647_READ_DIR  = Path(os.getenv("TOPIKII4647_READ_DIR",  "data/topik2/reading/46-47/OUT")).resolve()
TOPIKII4850_READ_DIR  = Path(os.getenv("TOPIKII4850_READ_DIR",  "data/topik2/reading/48-50/OUT")).resolve()

# ====== GIỚI HẠN CÂU THEO LEVEL ======
LEVEL_CAU_LIMITS = {"topik1": 70, "topik2": 50}

# ====== ĐIỂM MẶC ĐỊNH THEO LEVEL ======
DEFAULT_SCORE_BY_LEVEL = {"topik1": 3, "topik2": 2}

# ---- ROUTING RULES (data-driven) ----
RULES: Dict[tuple, Dict] = {
    ("topik1", "listening"): {
        "ranges": [
            {"cau_lo": 1,  "cau_hi": 4,  "dir": TOPIKI14_LISTEN_DIR,   "qno_required": False,
             "fallback_score_by_cau": {1: 4, 2: 4, 3: 3, 4: 3}},
            {"cau_lo": 5,  "cau_hi": 6,  "dir": TOPIKI56_LISTEN_DIR,   "qno_required": True,
             "fallback_score_by_cau": {5: 4, 6: 3}},
            {"cau_lo": 7,  "cau_hi": 10, "dir": TOPIKI710_LISTEN_DIR,  "qno_required": False,
             "fallback_score_by_cau": {7: 3, 8: 3, 9: 3, 10: 4}},
            {"cau_lo": 11, "cau_hi": 14, "dir": TOPIKI1114_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {11: 3, 12: 3, 13: 4, 14: 3}},
            {"cau_lo": 15, "cau_hi": 16, "dir": TOPIKI1516_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {15: 4, 16: 4}},
            {"cau_lo": 17, "cau_hi": 21, "dir": TOPIKI1721_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {17: 3, 18: 3, 19: 3, 20: 3, 21: 3}},
            {"cau_lo": 22, "cau_hi": 24, "dir": TOPIKI2224_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {22: 3, 23: 3, 24: 3}},
            {"cau_lo": 25, "cau_hi": 26, "dir": TOPIKI2526_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {25: 3, 26: 4}},
            {"cau_lo": 27, "cau_hi": 28, "dir": TOPIKI2728_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {27: 3, 28: 4}},
            {"cau_lo": 29, "cau_hi": 30, "dir": TOPIKI2930_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {29: 3, 30: 4}},
        ]
    },
    ("topik1", "reading"): {
        "ranges": [
            {"cau_lo": 31, "cau_hi": 33, "dir": TOPIKI3133_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {31: 2, 32: 2, 33: 2}},
            {"cau_lo": 34, "cau_hi": 39, "dir": TOPIKI3439_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {34: 2, 35: 2, 36: 2, 37: 3, 38: 3, 39: 2}},
            {"cau_lo": 40, "cau_hi": 42, "dir": TOPIKI4042_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {40: 2, 41: 2, 42: 2}},
            {"cau_lo": 43, "cau_hi": 45, "dir": TOPIKI4345_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {43: 3, 44: 2, 45: 3}},
            {"cau_lo": 46, "cau_hi": 48, "dir": TOPIKI4648_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {46: 3, 47: 3, 48: 2}},
            {"cau_lo": 49, "cau_hi": 56, "dir": TOPIKI4956_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {49: 2, 50: 2, 51: 3, 52: 2, 53: 2, 54: 3, 55: 2, 56: 3}},
            {"cau_lo": 57, "cau_hi": 58, "dir": TOPIKI5758_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {57: 3, 58: 2}},
            {"cau_lo": 59, "cau_hi": 62, "dir": TOPIKI5962_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {59: 2, 60: 3, 61: 2, 62: 2}},
            {"cau_lo": 63, "cau_hi": 64, "dir": TOPIKI6364_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {63: 2, 64: 3}},
            {"cau_lo": 65, "cau_hi": 70, "dir": TOPIKI6570_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {65: 2, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3}},
        ]
    },
    ("topik2", "listening"): {
        "ranges": [
            {"cau_lo": 1,  "cau_hi": 3,  "dir": TOPIKII13_LISTEN_DIR,   "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(1, 4)}},
            {"cau_lo": 4,  "cau_hi": 8,  "dir": TOPIKII48_LISTEN_DIR,   "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(4, 9)}},
            {"cau_lo": 9,  "cau_hi": 12, "dir": TOPIKII912_LISTEN_DIR,  "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(9, 13)}},
            {"cau_lo": 13, "cau_hi": 16, "dir": TOPIKII1316_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(13, 17)}},
            {"cau_lo": 17, "cau_hi": 20, "dir": TOPIKII1720_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(17, 21)}},
            {"cau_lo": 21, "cau_hi": 30, "dir": TOPIKII2130_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(21, 31)}},
            {"cau_lo": 31, "cau_hi": 32, "dir": TOPIKII3132_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {31: 2, 32: 2}},
            {"cau_lo": 33, "cau_hi": 36, "dir": TOPIKII3336_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(33, 37)}},
            {"cau_lo": 37, "cau_hi": 38, "dir": TOPIKII3738_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {37: 2, 38: 2}},
            {"cau_lo": 39, "cau_hi": 40, "dir": TOPIKII3940_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {39: 2, 40: 2}},
            {"cau_lo": 41, "cau_hi": 42, "dir": TOPIKII4142_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {41: 2, 42: 2}},
            {"cau_lo": 43, "cau_hi": 44, "dir": TOPIKII4344_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {43: 2, 44: 2}},
            {"cau_lo": 45, "cau_hi": 46, "dir": TOPIKII4546_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {45: 2, 46: 2}},
            {"cau_lo": 47, "cau_hi": 48, "dir": TOPIKII4748_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {47: 2, 48: 2}},
            {"cau_lo": 49, "cau_hi": 50, "dir": TOPIKII4950_LISTEN_DIR, "qno_required": False,
             "fallback_score_by_cau": {49: 2, 50: 2}},
        ]
    },
    ("topik2", "reading"): {
        "ranges": [
            {"cau_lo": 1,  "cau_hi": 2,  "dir": TOPIKII12_READ_DIR,   "qno_required": False,
             "fallback_score_by_cau": {1: 2, 2: 2}},
            {"cau_lo": 3,  "cau_hi": 4,  "dir": TOPIKII34_READ_DIR,   "qno_required": False,
             "fallback_score_by_cau": {3: 2, 4: 2}},
            {"cau_lo": 5,  "cau_hi": 8,  "dir": TOPIKII58_READ_DIR,   "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(5, 9)}},
            {"cau_lo": 9,  "cau_hi": 12, "dir": TOPIKII912_READ_DIR,  "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(9, 13)}},
            {"cau_lo": 13, "cau_hi": 15, "dir": TOPIKII1315_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {13: 2, 14: 2, 15: 2}},
            {"cau_lo": 16, "cau_hi": 18, "dir": TOPIKII1618_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {16: 2, 17: 2, 18: 2}},
            {"cau_lo": 19, "cau_hi": 22, "dir": TOPIKII1922_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(19, 23)}},
            {"cau_lo": 23, "cau_hi": 24, "dir": TOPIKII2324_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {23: 2, 24: 2}},
            {"cau_lo": 25, "cau_hi": 27, "dir": TOPIKII2527_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {25: 2, 26: 2, 27: 2}},
            {"cau_lo": 28, "cau_hi": 31, "dir": TOPIKII2831_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(28, 32)}},
            {"cau_lo": 32, "cau_hi": 34, "dir": TOPIKII3234_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {32: 2, 33: 2, 34: 2}},
            {"cau_lo": 35, "cau_hi": 38, "dir": TOPIKII3538_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {q: 2 for q in range(35, 39)}},
            {"cau_lo": 39, "cau_hi": 41, "dir": TOPIKII3941_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {39: 2, 40: 2, 41: 2}},
            {"cau_lo": 42, "cau_hi": 43, "dir": TOPIKII4243_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {42: 2, 43: 2}},
            {"cau_lo": 44, "cau_hi": 45, "dir": TOPIKII4445_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {44: 2, 45: 2}},
            {"cau_lo": 46, "cau_hi": 47, "dir": TOPIKII4647_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {46: 2, 47: 2}},
            {"cau_lo": 48, "cau_hi": 50, "dir": TOPIKII4850_READ_DIR, "qno_required": False,
             "fallback_score_by_cau": {48: 2, 49: 2, 50: 2}},
        ]
    },
}

# Pairs to group (single output with 2 items)
def _to_group_map(groups: Iterable[Iterable[int]]) -> Dict[int, Tuple[int, ...]]:
    """
    Nhận vào tập các group (mỗi group có >=2 câu), trả về map: qno -> tuple(sorted(group))
    Ví dụ: {(63,64), (48,49,50)} -> {63:(63,64), 64:(63,64), 48:(48,49,50), 49:(48,49,50), 50:(48,49,50)}
    """
    idx: Dict[int, Tuple[int, ...]] = {}
    for g in groups:
        if not isinstance(g, (tuple, list, set)):
            raise ValueError(f"Invalid group specification: {g!r}")
        s = sorted({int(x) for x in g})
        if len(s) < 2:
            continue
        t = tuple(s)
        for q in t:
            idx[q] = t
    return idx

# Các cặp đi theo đề TOPIK chuẩn
GROUPS_RAW = {
    # TOPIK I — Listening: nhóm tranh/hội thoại 25–26, 27–28, 29–30
    ("topik1", "listening"): {(25, 26), (27, 28), (29, 30)},

    # TOPIK I — Reading: hầu hết là đoạn văn 2 câu theo cặp từ 49–70
    ("topik1", "reading"): {
        (49, 50), (51, 52), (53, 54), (55, 56), 
        (59, 60), (61, 62), (63, 64), (65, 66),
         (67, 68), (69, 70),
    },

    # TOPIK II — nếu bạn có cặp, bổ sung tại đây; mặc định để trống
    ("topik2", "listening"): {
        (21, 22), (23, 24), (25, 26), (27, 28),
        (29, 30), (31, 32), (33, 34), (35, 36),
        (37, 38), (39, 40), (41, 42), (43, 44),
        (45, 46), (47, 48), (49, 50)
    },
    ("topik2", "reading"): {
        (19, 20), (21, 22), (23, 24), (42, 43),
        (44, 45), (46, 47), (48, 49, 50)
    },
}

# Chuẩn hoá thành map 2 chiều để dùng trực tiếp
GROUP_MAPS = {k: _to_group_map(v) for k, v in GROUPS_RAW.items()}

def get_group_map(level: str, danh_muc: str) -> Dict[int, Tuple[int, ...]]:
    key = (norm(level), norm(danh_muc))
    return GROUP_MAPS.get(key, {})

def get_group_for(level: str, danh_muc: str, qno: int) -> Optional[Tuple[int, ...]]:
    return get_group_map(level, danh_muc).get(int(qno))