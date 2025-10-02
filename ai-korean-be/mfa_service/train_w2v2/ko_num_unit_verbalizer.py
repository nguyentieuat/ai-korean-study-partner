#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ko_num_unit_verbalizer.py

Reusable Korean number & unit verbalizer.

- Converts numerals and symbols (25%, 3km, 10s, 35°C, $2.5) to spoken Korean.
- Supports Sino-Korean by default; optionally uses Native Korean for counters
  like 시/시간/달/살/번/개/명/... with correct bound forms (한/두/세/네/스무).
- Designed to be imported in multiple pipelines (MFA .lab generation, ASR/CTC targets).
- Also provides a tiny CLI: pipe lines via stdin to get verbalized lines.

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations
import re
import sys
from typing import Iterable

# ---------- Sino-Korean helpers ----------

_SINO_DIGITS = ["영","일","이","삼","사","오","육","칠","팔","구"]
_SINO_SMALL = [(1000,"천"),(100,"백"),(10,"십")]
_SINO_BIG   = [(10**12,"조"),(10**8,"억"),(10**4,"만")]

def _sino_int(n: int) -> str:
    """Read non-negative integer in Sino-Korean (e.g., 123 -> 백 이십 삼)."""
    if n == 0:
        return "영"
    out = []
    def _chunk_to_sino(x: int) -> str:
        if x == 0:
            return ""
        parts = []
        for val, name in _SINO_SMALL:
            q, x = divmod(x, val)
            if q:
                parts.append(name if q == 1 else _SINO_DIGITS[q] + " " + name)
        if x:
            parts.append(_SINO_DIGITS[x])
        return " ".join(parts).strip()
    for val, name in _SINO_BIG:
        q, n = divmod(n, val)
        if q:
            out.append((_chunk_to_sino(q) + " " + name).strip())
    if n:
        out.append(_chunk_to_sino(n))
    return " ".join(out).strip()

def _sino_any(num_str: str) -> str:
    """Read an integer or decimal (with dot) in Sino-Korean."""
    s = num_str.replace(",", "")
    if s.startswith("."):
        s = "0" + s
    if "." in s:
        a, b = s.split(".", 1)
        a_read = _sino_int(int(a)) if a else "영"
        b_read = " ".join(_SINO_DIGITS[int(ch)] for ch in b if ch.isdigit())
        return f"{a_read} 점 {b_read}".strip()
    return _sino_int(int(s))

# ---------- Native-Korean helpers (1~99) ----------

_NATIVE_UNITS = ["","하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
_NATIVE_UNITS_BOUND = {1:"한", 2:"두", 3:"세", 4:"네"}  # bound forms before counters
_NATIVE_TENS = {10:"열", 20:"스물", 30:"서른", 40:"마흔", 50:"쉰", 60:"예순", 70:"일흔", 80:"여든", 90:"아흔"}

def native_1_99(n: int, bound: bool = True) -> str:
    """Native-Korean reading for 1..99, using bound forms (한/두/세/네/스무) when bound=True."""
    assert 1 <= n <= 99
    if n < 10:
        if bound and n in _NATIVE_UNITS_BOUND:
            return _NATIVE_UNITS_BOUND[n]
        return _NATIVE_UNITS[n]
    if n == 20:
        return "스무" if bound else "스물"
    tens = (n // 10) * 10
    ones = n % 10
    base = _NATIVE_TENS[tens]
    if ones == 0:
        return "스무" if (bound and n == 20) else base
    if bound and ones in _NATIVE_UNITS_BOUND:
        ones_s = _NATIVE_UNITS_BOUND[ones]
    else:
        ones_s = _NATIVE_UNITS[ones]
    return f"{base}{ones_s}" if tens != 20 else f"{('스무' if bound else '스물')}{ones_s}"

# ---------- Main verbalizer ----------

_NUM_TOKEN_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

# Precompile some unit patterns (order matters!)
_RE_PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_RE_SPEED   = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:km/h|kph)\b", re.IGNORECASE)
_RE_TEMP_F  = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*°?\s*F\b", re.IGNORECASE)
_RE_TEMP_K  = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*K\b", re.IGNORECASE)
_RE_TEMP_C  = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:°|℃)?\s*C\b", re.IGNORECASE)


# Metric length (ordered: km, cm, mm, then standalone m)
_RE_KM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*km\b", re.IGNORECASE)
_RE_CM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*cm\b", re.IGNORECASE)
_RE_MM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*mm\b", re.IGNORECASE)
_RE_M  = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*m\b")

# Mass
_RE_KG = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*kg\b", re.IGNORECASE)
_RE_G  = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*g\b",  re.IGNORECASE)

# Time shorthands (Latin) map to Sino by default
_RE_H = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*h(?:ours?|r)?\b", re.IGNORECASE)
_RE_MIN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*min(?:ute)?s?\b", re.IGNORECASE)
_RE_S = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b", re.IGNORECASE)
_RE_MS = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*ms\b", re.IGNORECASE)

# Currency
_RE_WON = re.compile(r"₩\s?(\d[\d,\.]*)")
_RE_USD = re.compile(r"\$\s?(\d[\d,\.]*)")
_RE_EUR = re.compile(r"€\s?(\d[\d,\.]*)")

# Native counters (Korean)
_RE_SI = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*시\b")      # clock hour
_RE_SIGAN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*시간\b") # duration hours
_RE_DAL = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*달\b")     # months duration
_RE_SAL = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*살\b")     # age native
_RE_SE  = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*세\b")     # age sino
_RE_BEON = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*번\b")    # times
# generic native counters group
_NATIVE_CTRS = r"(개|명|사람|마리|권|잔|병|대)"
_RE_NATIVE_COUNTERS = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*{_NATIVE_CTRS}\b")

def _read_native_or_sino_int(num: str, bound=True) -> str:
    """Prefer Native 1..99, otherwise Sino; decimals -> Sino."""
    if "." in num:
        return _sino_any(num)
    n = int(num.replace(",", ""))
    if 1 <= n <= 99:
        return native_1_99(n, bound=bound)
    return _sino_int(n)

def verbalize_ko_numbers_units(
    text: str,
    use_native_counters: bool = True,
    include_celsius_prefix: bool = True,
) -> str:
    """
    Verbalize numbers & units into spoken Korean text.

    Args:
        text: input text.
        use_native_counters: if True, use Native Korean (한/두/세/네/스무...) for counters
            like 시/시간/달/살/번/개/명/사람/마리/권/잔/병/대 (1..99, else Sino).
        include_celsius_prefix: if True, include '섭씨' before temperature degree.

    Returns:
        str: verbalized text.
    """
    if not text:
        return text
    t = text

    # ----- Native counters first (if enabled) -----
    if use_native_counters:
        t = _RE_SI.sub(lambda m: f"{_read_native_or_sino_int(m.group(1), True)} 시", t)

        def _hours(m):
            s = m.group(1).replace(",", "")
            if "." in s:
                try:
                    v = float(s)
                    iv, fv = int(v), abs(v - int(v))
                    if abs(fv - 0.5) < 1e-8 and 1 <= iv <= 99:
                        return f"{native_1_99(iv, True)} 시간 반"
                except Exception:
                    pass
                return f"{_sino_any(s)} 시간"
            n = int(s)
            return f"{native_1_99(n, True) if 1<=n<=99 else _sino_int(n)} 시간"
        t = _RE_SIGAN.sub(_hours, t)

        def _months(m):
            s = m.group(1).replace(",", "")
            if "." in s:
                try:
                    v = float(s)
                    iv, fv = int(v), abs(v - int(v))
                    if abs(fv - 0.5) < 1e-8 and 1 <= iv <= 99:
                        return f"{native_1_99(iv, True)} 달 반"
                except Exception:
                    pass
                return f"{_sino_any(s)} 달"
            n = int(s)
            return f"{native_1_99(n, True) if 1<=n<=99 else _sino_int(n)} 달"
        t = _RE_DAL.sub(_months, t)

        def _age_sal(m):
            s = m.group(1).replace(",", "")
            if "." in s:
                try:
                    v = float(s)
                    iv, fv = int(v), abs(v - int(v))
                    if abs(fv - 0.5) < 1e-8 and 1 <= iv <= 99:
                        return f"{native_1_99(iv, True)} 살 반"
                except Exception:
                    pass
                return f"{_sino_any(s)} 살"
            n = int(s)
            return f"{native_1_99(n, True) if 1<=n<=99 else _sino_int(n)} 살"
        t = _RE_SAL.sub(_age_sal, t)

        t = _RE_SE.sub(lambda m: f"{_sino_any(m.group(1))} 세", t)
        t = _RE_BEON.sub(lambda m: f"{_read_native_or_sino_int(m.group(1), True)} 번", t)
        t = _RE_NATIVE_COUNTERS.sub(lambda m: f"{_read_native_or_sino_int(m.group(1), True)} {m.group(2)}", t)

    # ----- Units that should stay Sino -----
    t = _RE_PERCENT.sub(lambda m: f"{_sino_any(m.group(1))} 퍼센트", t)
    t = _RE_SPEED.sub(lambda m: f"시속 {_sino_any(m.group(1))} 킬로미터", t)
    t = _RE_TEMP_C.sub(lambda m: f"{'섭씨 ' if include_celsius_prefix else ''}{_sino_any(m.group(1))} 도", t)
    t = _RE_TEMP_F.sub(lambda m: f"화씨 {_sino_any(m.group(1))} 도", t)  
    t = _RE_TEMP_K.sub(lambda m: f"{_sino_any(m.group(1))} 켈빈", t)       

    # length (careful ordering)
    t = _RE_KM.sub(lambda m: f"{_sino_any(m.group(1))} 킬로미터", t)
    t = _RE_CM.sub(lambda m: f"{_sino_any(m.group(1))} 센티미터", t)
    t = _RE_MM.sub(lambda m: f"{_sino_any(m.group(1))} 밀리미터", t)
    t = _RE_M.sub(lambda m: f"{_sino_any(m.group(1))} 미터", t)

    # mass
    t = _RE_KG.sub(lambda m: f"{_sino_any(m.group(1))} 킬로그램", t)
    t = _RE_G.sub(lambda m: f"{_sino_any(m.group(1))} 그램", t)

    # time shorthands (Latin)
    t = _RE_H.sub(lambda m: f"{_sino_any(m.group(1))} 시간", t)
    t = _RE_MIN.sub(lambda m: f"{_sino_any(m.group(1))} 분", t)
    t = _RE_S.sub(lambda m: f"{_sino_any(m.group(1))} 초", t)
    t = _RE_MS.sub(lambda m: f"{_sino_any(m.group(1))} 밀리초", t)

    # currency
    t = _RE_WON.sub(lambda m: f"{_sino_any(m.group(1))} 원", t)
    t = _RE_USD.sub(lambda m: f"{_sino_any(m.group(1))} 달러", t)
    t = _RE_EUR.sub(lambda m: f"{_sino_any(m.group(1))} 유로", t)

    # Remaining bare numbers -> Sino
    t = _NUM_TOKEN_RE.sub(lambda m: _sino_any(m.group(0)), t)

    # whitespace cleanup
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- Small CLI ----------

def _cli(stdin: Iterable[str], use_native_counters: bool = True, include_celsius_prefix: bool = True) -> int:
    for line in stdin:
        line = line.rstrip("\n")
        print(verbalize_ko_numbers_units(line, use_native_counters=use_native_counters,
                                         include_celsius_prefix=include_celsius_prefix))
    return 0

if __name__ == "__main__":
    # Minimal flags: --no-native, --no-celsius
    use_native = True
    with_celsius = True
    args = [a.lower() for a in sys.argv[1:]]
    if "--no-native" in args:
        use_native = False
    if "--no-celsius" in args:
        with_celsius = False
    sys.exit(_cli(sys.stdin, use_native_counters=use_native, include_celsius_prefix=with_celsius))
