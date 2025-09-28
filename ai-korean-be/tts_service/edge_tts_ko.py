#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
edge_tts_ko.py (v1.3)
- Align with edge-tts validation as of your env:
  * rate: signed percent, e.g., "+0%"
  * volume: signed percent, e.g., "+0%"
  * pitch: signed Hz, e.g., "+0Hz" (NOT "st")
- Sanitizer now coerces missing sign/units.
- Accepts legacy inputs like "0%", "0Hz", and fixes them.
- If user passes semitone (e.g., "+2st"), we fallback to "+0Hz" and warn.
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Optional

import edge_tts
import aiofiles


DEFAULT_FEMALE = "ko-KR-SunHiNeural"
DEFAULT_MALE   = "ko-KR-InJoonNeural"


def _fmt_voice(v: dict) -> str:
    short = v.get("ShortName", "")
    gender = v.get("Gender", "")
    locale = v.get("Locale", "")
    friendly = v.get("FriendlyName") or v.get("DisplayName") or short
    return f"{short:<24} | {gender:<6} | {locale} | {friendly}"


async def list_voices(show_all: bool = False) -> None:
    voices = await edge_tts.list_voices()
    if show_all:
        for v in voices:
            print(_fmt_voice(v))
    else:
        for v in voices:
            if v.get("Locale") == "ko-KR":
                print(_fmt_voice(v))


def pick_voice(explicit: Optional[str], want_male: bool, want_female: bool, available: List[dict]) -> str:
    if explicit:
        return explicit
    preferred = DEFAULT_MALE if want_male else DEFAULT_FEMALE
    shortnames = {v.get("ShortName") for v in available if v.get("ShortName")}
    if preferred in shortnames:
        return preferred
    for v in available:
        if v.get("Locale") == "ko-KR":
            if want_male and v.get("Gender") == "Male":
                return v.get("ShortName")
            if want_female and v.get("Gender") == "Female":
                return v.get("ShortName")
    for v in available:
        if v.get("Locale") == "ko-KR":
            return v.get("ShortName")
    raise RuntimeError("No Korean voices found. Try --list-voices --all to inspect availability.")


def _ensure_signed_percent(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "+0%"
    if v.endswith("st"):
        # Library in your env does not accept semitones; warn and neutralize.
        print("[WARN] Semitone unit 'st' not supported in this edge-tts build. Using +0%.")
        return "+0%"
    # normalize sign + unit '%'
    if v[0] in "+-":
        return v if v.endswith("%") else (v + "%")
    # plain number
    return f"+{v}%"


def _ensure_signed_hz(value: str) -> str:
    v = (value or "").strip()
    if not v:
        return "+0Hz"
    if v.endswith("st"):
        # Can't convert reliably without base freq
        print("[WARN] Semitone unit 'st' not supported here; using +0Hz.")
        return "+0Hz"
    if v[0] in "+-":
        return v if v.endswith("Hz") else (v + "Hz")
    return f"+{v}Hz"


async def synth_one(text: str, out_path: Path, voice: str, rate: str, pitch: str, volume: str) -> None:
    rate   = _ensure_signed_percent(rate)   # e.g., "-10%"
    volume = _ensure_signed_percent(volume) # e.g., "+2%"
    pitch  = _ensure_signed_hz(pitch)       # e.g., "+0Hz"
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(out_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                await f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass


async def synth_from_text_file(text_file: Path, out_path: Path, voice: str, rate: str, pitch: str, volume: str) -> None:
    text = Path(text_file).read_text(encoding="utf-8")
    await synth_one(text.strip(), out_path, voice, rate, pitch, volume)


async def synth_from_lines_file(lines_file: Path, out_dir: Path, voice: str, rate: str, pitch: str, volume: str) -> None:
    lines = Path(lines_file).read_text(encoding="utf-8").splitlines()
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = len(str(len(lines)))
    tasks = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        fname = f"{i:0{pad}d}.mp3"
        tasks.append(synth_one(line, out_dir / fname, voice, rate, pitch, volume))
    await asyncio.gather(*tasks)


async def main():
    ap = argparse.ArgumentParser(description="Edge TTS (Korean) helper")
    ap.add_argument("-t", "--text", help="Text to synthesize (single utterance)")
    ap.add_argument("-o", "--out", help="Output file path (e.g., out.mp3)")
    ap.add_argument("--text-file", help="Read a whole text file to synthesize", type=str)
    ap.add_argument("--lines-file", help="Read many lines; each line -> one mp3", type=str)
    ap.add_argument("--out-dir", help="Output directory for --lines-file mode", type=str)

    ap.add_argument("--voice", help="Explicit voice short name (e.g., ko-KR-InJoonNeural)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--male", action="store_true", help="Prefer a male Korean voice")
    g.add_argument("--female", action="store_true", help="Prefer a female Korean voice (default)")

    ap.add_argument("--rate", default="+0%", help='Speaking rate change (percent), e.g., "-15%" or "+10%"')
    ap.add_argument("--pitch", default="+0Hz", help='Pitch change (Hz), e.g., "+0Hz", "-20Hz"')
    ap.add_argument("--volume", default="+0%", help='Volume change (percent), e.g., "+2%", "-3%"')

    ap.add_argument("--list-voices", action="store_true", help="List available voices (ko-KR only by default)")
    ap.add_argument("--all", action="store_true", help="When listing voices, show ALL locales")

    args = ap.parse_args()

    if args.list_voices:
        await list_voices(show_all=args.all)
        return

    mode_count = sum(bool(x) for x in [args.text, args.text_file, args.lines_file])
    if mode_count != 1:
        raise SystemExit("Please specify exactly ONE of: --text, --text-file, or --lines-file")

    voices = await edge_tts.list_voices()
    voice = pick_voice(args.voice, args.male, args.female or not args.male, voices)

    if args.text:
        if not args.out:
            raise SystemExit("Please specify --out for single text mode (e.g., out.mp3)")
        await synth_one(args.text, Path(args.out), voice, args.rate, args.pitch, args.volume)
        print(f"[OK] Wrote: {args.out} (voice={voice})")
    elif args.text_file:
        if not args.out:
            raise SystemExit("Please specify --out for --text-file mode (e.g., out.mp3)")
        await synth_from_text_file(Path(args.text_file), Path(args.out), voice, args.rate, args.pitch, args.volume)
        print(f"[OK] Wrote: {args.out} (voice={voice}) from {args.text_file}")
    elif args.lines_file:
        if not args.out_dir:
            raise SystemExit("Please specify --out-dir for --lines-file mode (folder to store mp3s)")
        await synth_from_lines_file(Path(args.lines_file), Path(args.out_dir), voice, args.rate, args.pitch, args.volume)
        print(f"[OK] Wrote batch mp3s to: {args.out_dir} (voice={voice}) from {args.lines_file}")


if __name__ == "__main__":
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    asyncio.run(main())
