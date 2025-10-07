#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chuyển JSONL thành nhiều file đơn theo key:
  key = sha1( base64(raw_bytes) )
KHÔNG re-encode ảnh trước khi hash.

- Input:  JSONL với "image": "36_40.png" (hoặc dataURL, hoặc http nếu bật)
- Output: Nhiều file JSON: <out_dir>/ab/cd/<key>.json
         (ab/cd là 2 cấp sharding từ 4 kí tự đầu của key)

Mặc định giữ nguyên record và chỉ bổ sung:
    "image_key": "<sha1>"
Tùy chọn:
  --emit-b64        : chèn "image_b64" vào file JSON (payload tự đủ)
  --emit-meta       : thêm meta như {"mime": "...", "size": N} nếu xác định được
  --allow-http      : cho phép tải ảnh http(s) (cần requests)
  --strict          : lỗi là dừng; mặc định: skip record lỗi
  --dry-run         : chỉ thống kê, không ghi file
  --flat            : KHÔNG sharding; ghi thẳng <out_dir>/<key>.json (không khuyến nghị khi nhiều file)

Ví dụ:
  python jsonl_to_keyfiles.py vision2text_6364.jsonl \
      --images-dir images \
      --out-dir data/vision_records \
      --emit-b64 --emit-meta
"""

import os, sys, json, base64, hashlib, argparse, mimetypes
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import urlparse

try:
    import requests  # chỉ dùng nếu --allow-http
except Exception:
    requests = None

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


# ----------------- utils: b64 & hashing -----------------
def sha1_of_b64(b64_str: str) -> str:
    h = hashlib.sha1(); h.update(b64_str.encode("utf-8")); return h.hexdigest()

def bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# ----------------- utils: path & IO -----------------
def atomic_write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)

def shard_path(root: Path, key: str, flat: bool) -> Path:
    if flat:
        return root / f"{key}.json"
    a, b = key[:2], key[2:4]
    return root / a / b / f"{key}.json"

def find_image_file(name_or_path: str, search_dirs: List[Path]) -> Optional[Path]:
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return p
    for d in search_dirs:
        cand = d / name_or_path
        if cand.exists() and cand.is_file():
            return cand
        if cand.suffix and cand.suffix.lower() != cand.suffix:
            cand2 = cand.with_suffix(cand.suffix.lower())
            if cand2.exists() and cand2.is_file():
                return cand2
    return None


# ----------------- utils: source detection -----------------
def is_data_url(s: str) -> bool:
    return s.startswith("data:image/")

def is_http_url(s: str) -> bool:
    try:
        u = urlparse(s); return u.scheme in ("http", "https")
    except Exception:
        return False

def data_url_to_b64(s: str) -> Optional[str]:
    try:
        return s.split(",", 1)[1]
    except Exception:
        return None

def sniff_mime_from_path(p: Path) -> Optional[str]:
    mime, _ = mimetypes.guess_type(str(p))
    return mime

def load_image_b64(image: str,
                   images_dirs: List[Path],
                   allow_http: bool = False) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """
    Trả về (b64, source_hint, size_bytes, mime)
    - b64: base64 không prefix data:
    - source_hint: file path hoặc url cho log
    - size_bytes: kích thước nguồn (nếu biết)
    - mime: nếu suy ra được
    """
    if not image or not isinstance(image, str):
        return None, None, None, None

    s = image.strip()

    # dataURL
    if is_data_url(s):
        b64 = data_url_to_b64(s)
        return b64, "dataurl", (len(b64) * 3 // 4 if b64 else None), None

    # local file or relative name
    if not is_http_url(s):
        f = find_image_file(s, images_dirs) if images_dirs else Path(s)
        if f and f.exists():
            raw = None
            try:
                raw = f.read_bytes()
            except Exception:
                pass
            if raw is not None:
                return bytes_to_b64(raw), str(f), len(raw), sniff_mime_from_path(f)
        return None, (str(f) if f else s), None, None

    # http(s)
    if is_http_url(s):
        if not allow_http or requests is None:
            return None, s, None, None
        try:
            r = requests.get(s, timeout=60)
            r.raise_for_status()
            b = r.content
            mime = r.headers.get("Content-Type")
            return bytes_to_b64(b), s, len(b), mime
        except Exception:
            return None, s, None, None

    return None, None, None, None


# ----------------- core -----------------
def process_jsonl_to_keyfiles(in_path: Path,
                              out_dir: Path,
                              images_dirs: List[Path],
                              emit_b64: bool,
                              emit_meta: bool,
                              allow_http: bool,
                              strict: bool,
                              flat: bool,
                              dry_run: bool) -> Tuple[int, int]:
    """
    Đọc JSONL và ghi nhiều file <key>.json dưới out_dir[/ab/cd].
    Return (ok_count, skip_count)
    """
    ok = 0
    skip = 0

    out_dir.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                skip += 1
                if strict:
                    raise RuntimeError(f"Line {ln}: JSON parse error: {e}")
                continue

            image = obj.get("image")
            if not isinstance(image, str):
                skip += 1
                if strict:
                    raise RuntimeError(f"Line {ln}: field 'image' must be a string")
                continue

            b64, hint, size_bytes, mime = load_image_b64(image, images_dirs, allow_http=allow_http)
            if not b64:
                skip += 1
                if strict:
                    raise RuntimeError(f"Line {ln}: Cannot load image '{image}' (hint={hint})")
                continue

            key = sha1_of_b64(b64)
            obj_out = dict(obj)  # clone
            obj_out["image_key"] = key
            if emit_b64:
                obj_out["image_b64"] = b64
            if emit_meta:
                meta = {}
                if size_bytes is not None: meta["size"] = size_bytes
                if mime: meta["mime"] = mime
                if meta:
                    obj_out["image_meta"] = meta

            out_path = shard_path(out_dir, key, flat=flat)
            if not dry_run:
                atomic_write_text(out_path, json.dumps(obj_out, ensure_ascii=False, indent=2))
            ok += 1

    return ok, skip


def main():
    ap = argparse.ArgumentParser(description="Split JSONL into per-key JSON files (<key>.json) using sha1(base64(raw_bytes)).")
    ap.add_argument("jsonl", type=str, help="Đường dẫn JSONL đầu vào.")
    ap.add_argument("--images-dir", action="append", default=[], help="Thư mục chứa ảnh (có thể lặp nhiều lần).")
    ap.add_argument("--out-dir", type=str, required=True, help="Thư mục đầu ra để ghi <key>.json (sharded).")
    ap.add_argument("--emit-b64", action="store_true", help="Nhúng 'image_b64' vào file JSON.")
    ap.add_argument("--emit-meta", action="store_true", help="Thêm 'image_meta' (mime/size) nếu xác định được.")
    ap.add_argument("--allow-http", action="store_true", help="Cho phép tải ảnh http(s) (cần requests).")
    ap.add_argument("--strict", action="store_true", help="Bật chế độ strict (lỗi sẽ dừng).")
    ap.add_argument("--flat", action="store_true", help="Không sharding thư mục; ghi <out_dir>/<key>.json trực tiếp.")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ chạy thống kê, không ghi file.")
    args = ap.parse_args()

    in_path = Path(args.jsonl)
    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    images_dirs = [Path(d) for d in args.images_dir]

    ok, skip = process_jsonl_to_keyfiles(
        in_path=in_path,
        out_dir=out_dir,
        images_dirs=images_dirs,
        emit_b64=args.emit_b64,
        emit_meta=args.emit_meta,
        allow_http=args.allow_http,
        strict=args.strict,
        flat=args.flat,
        dry_run=args.dry_run,
    )

    mode = "DRY-RUN" if args.dry_run else "WROTE"
    print(f"[{mode}] out_dir={out_dir} | ok={ok}, skip={skip}")


if __name__ == "__main__":
    main()