# generate_question/picker.py
from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path

from .cache import ensure_ready, choose_nonrepeating, choose_pair_by_group, choose_group_by_qnos, _DIR_CACHE
__all__ = ["ensure_ready", "choose_nonrepeating", "choose_pair_by_group", "choose_group_by_qnos", "DIR_CACHE"]

# re-export a read-only view if you want
DIR_CACHE = _DIR_CACHE