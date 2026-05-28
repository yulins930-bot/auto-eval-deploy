# -*- coding: utf-8 -*-
"""学生作答图片：CSV/JSON 内 URL 与 Excel 单元格内嵌图。"""
from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_JSON_IMAGE_URL_RE = re.compile(r'"imageUrl"\s*:\s*"([^"]+)"', re.I)
_PLAIN_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)

_WB_CACHE: Dict[str, Any] = {}
_WB_LOCK = threading.Lock()
_IMAGE_INDEX_CACHE: Dict[str, Dict[tuple[int, int], Tuple[bytes, str]]] = {}

_IMAGE_MAX_EDGE = 1600
_JPEG_QUALITY = 85


def cell_value_looks_like_image_url(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return False
    if _PLAIN_URL_RE.search(s) or _JSON_IMAGE_URL_RE.search(s):
        return True
    if s.startswith("{") or s.startswith("["):
        try:
            return bool(extract_image_urls_from_value(json.loads(s)))
        except json.JSONDecodeError:
            pass
    return False


def extract_image_urls_from_value(value: Any) -> List[str]:
    urls: List[str] = []
    if value is None:
        return urls
    if isinstance(value, dict):
        for key in ("imageUrl", "image_url", "url", "img_url", "answer_img_url"):
            u = value.get(key)
            if u and isinstance(u, str) and u.startswith("http"):
                urls.append(u.strip())
        for v in value.values():
            urls.extend(extract_image_urls_from_value(v))
        return _dedupe_urls(urls)
    if isinstance(value, list):
        for item in value:
            urls.extend(extract_image_urls_from_value(item))
        return _dedupe_urls(urls)
    s = str(value).strip()
    if not s:
        return urls
    for m in _JSON_IMAGE_URL_RE.finditer(s):
        urls.append(m.group(1).strip())
    if not urls:
        m = _PLAIN_URL_RE.search(s)
        if m:
            urls.append(m.group(0).strip().rstrip(".,;)]}"))
    return _dedupe_urls(urls)


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_image_bytes(url: str, *, timeout: int = 60) -> Tuple[bytes, str]:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; BatchEval/1.0)"}
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()
    raw = r.content
    mime = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        mime = _guess_mime(raw)
    return _maybe_compress(raw), mime


def _guess_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _maybe_compress(data: bytes) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return data
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        max_e = _IMAGE_MAX_EDGE
        if max(w, h) > max_e:
            ratio = max_e / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        return buf.getvalue()
    except Exception:
        return data


def detect_embedded_image_columns(path: Path, columns: List[str]) -> List[str]:
    if path.suffix.lower() != ".xlsx":
        return []
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        ws = wb.active
        col_names = list(columns)
        n = len(col_names)
        cols_hit: set[str] = set()
        for img in getattr(ws, "_images", []) or []:
            for ci in _anchor_column_span(img, n):
                if 0 <= ci < n and col_names[ci]:
                    cols_hit.add(col_names[ci])
        return sorted(cols_hit)
    finally:
        wb.close()


def _anchor_column_span(img: Any, ncols: int) -> List[int]:
    """内嵌图可能跨多列：返回与锚点相交的列索引（0-based）。"""
    try:
        c0 = int(img.anchor._from.col)
        if hasattr(img.anchor, "_to") and img.anchor._to is not None:
            c1 = int(img.anchor._to.col)
        else:
            c1 = c0
        lo, hi = min(c0, c1), max(c0, c1)
        return [c for c in range(lo, hi + 1) if 0 <= c < ncols]
    except Exception:
        return []


def _cell_covered_by_image_anchor(img: Any, target_row: int, target_col: int) -> bool:
    """判断 (target_row,target_col) 是否落在图片锚点范围内（支持 TwoCellAnchor 跨格）。"""
    try:
        r0, c0 = int(img.anchor._from.row), int(img.anchor._from.col)
        if hasattr(img.anchor, "_to") and img.anchor._to is not None:
            r1, c1 = int(img.anchor._to.row), int(img.anchor._to.col)
        else:
            r1, c1 = r0, c0
        r_lo, r_hi = min(r0, r1), max(r0, r1)
        c_lo, c_hi = min(c0, c1), max(c0, c1)
        return r_lo <= target_row <= r_hi and c_lo <= target_col <= c_hi
    except Exception:
        return False


def _get_workbook(path: Path):
    import openpyxl

    key = str(path.resolve())
    with _WB_LOCK:
        if key not in _WB_CACHE:
            _WB_CACHE[key] = openpyxl.load_workbook(path, data_only=True)
        return _WB_CACHE[key]


def build_excel_image_index(path: Path, columns: List[str]) -> Dict[tuple[int, int], Tuple[bytes, str]]:
    """一次性扫描 Excel 内嵌图，建立 (row0, col0) → (bytes, mime) 索引，避免每行全表扫描。"""
    if path.suffix.lower() != ".xlsx":
        return {}
    key = str(path.resolve())
    with _WB_LOCK:
        cached = _IMAGE_INDEX_CACHE.get(key)
        if cached is not None:
            return cached

    index: Dict[tuple[int, int], Tuple[bytes, str]] = {}
    wb = _get_workbook(path)
    ws = wb.active
    col_names = list(columns)
    n = len(col_names)
    for img in getattr(ws, "_images", []) or []:
        try:
            r0 = int(img.anchor._from.row)
            c0 = int(img.anchor._from.col)
            if hasattr(img.anchor, "_to") and img.anchor._to is not None:
                r1, c1 = int(img.anchor._to.row), int(img.anchor._to.col)
            else:
                r1, c1 = r0, c0
            data = None
            try:
                data = img._data()
            except Exception:
                pass
            if not data and hasattr(wb, "_archive") and wb._archive:
                img_path = getattr(img, "path", None)
                if img_path:
                    data = wb._archive.read(str(img_path).lstrip("/"))
            if not data:
                continue
            payload = (_maybe_compress(data), _guess_mime(data))
            for r in range(min(r0, r1), max(r0, r1) + 1):
                for c in range(min(c0, c1), max(c0, c1) + 1):
                    if 0 <= c < n:
                        index[(r, c)] = payload
        except Exception:
            continue
    with _WB_LOCK:
        _IMAGE_INDEX_CACHE[key] = index
    return index


def extract_image_from_excel_cell(
    path: Path,
    *,
    excel_row: int,
    column_name: str,
    columns: List[str],
) -> Optional[Tuple[bytes, str]]:
    if path.suffix.lower() != ".xlsx" or column_name not in columns:
        return None
    col_idx = columns.index(column_name)
    target_row = excel_row - 1
    target_col = col_idx
    wb = _get_workbook(path)
    ws = wb.active
    for img in getattr(ws, "_images", []) or []:
        try:
            if not _cell_covered_by_image_anchor(img, target_row, target_col):
                continue
            data = None
            try:
                data = img._data()
            except Exception:
                pass
            if not data and hasattr(wb, "_archive") and wb._archive:
                img_path = getattr(img, "path", None)
                if img_path:
                    data = wb._archive.read(str(img_path).lstrip("/"))
            if data:
                return _maybe_compress(data), _guess_mime(data)
        except Exception:
            continue
    return None


def infer_primary_content_mode(
    field_mapping: Dict[str, Any],
    row_dict: Dict[str, str],
    *,
    file_kind: str,
    embedded_image_columns: Optional[List[str]] = None,
) -> str:
    explicit = (field_mapping or {}).get("primary_content_mode") or ""
    if explicit in ("text", "image_url", "excel_embedded"):
        return explicit
    primary = (field_mapping or {}).get("primary_content") or ""
    if not primary:
        return "text"
    embedded = set(embedded_image_columns or [])
    if file_kind == "xlsx" and primary in embedded:
        return "excel_embedded"
    if cell_value_looks_like_image_url(row_dict.get(primary)):
        return "image_url"
    return "text"


def assess_job_vision(
    field_mapping: Dict[str, Any],
    *,
    file_kind: str,
    embedded_image_columns: List[str],
    columns: List[str],
    sample_rows: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    primary = (field_mapping or {}).get("primary_content") or ""
    mode = "text"
    required = False
    if primary:
        if file_kind == "xlsx" and primary in (embedded_image_columns or []):
            mode = "excel_embedded"
            required = True
        elif sample_rows and columns:
            idx = columns.index(primary) if primary in columns else -1
            if idx >= 0:
                for rv in sample_rows[:20]:
                    if idx < len(rv) and cell_value_looks_like_image_url(rv[idx]):
                        mode = "image_url"
                        required = True
                        break
        elif file_kind == "csv" and primary:
            low = primary.lower()
            if any(k in low for k in ("img", "url", "图", "picture", "image", "photo")):
                mode = "image_url"
                required = True
    return {
        "vision_required": required,
        "primary_content_mode": mode,
        "embedded_image_columns": list(embedded_image_columns or []),
        "primary_content": primary,
    }


def resolve_student_answer_images(
    row_index: int,
    row_dict: Dict[str, str],
    field_mapping: Dict[str, Any],
    columns: List[str],
    *,
    image_source_path: Optional[str],
    file_kind: str,
    embedded_image_columns: Optional[List[str]] = None,
    excel_image_index: Optional[Dict[tuple[int, int], Tuple[bytes, str]]] = None,
) -> List[Dict[str, Any]]:
    primary = (field_mapping or {}).get("primary_content") or ""
    if not primary:
        return []
    mode = infer_primary_content_mode(
        field_mapping,
        row_dict,
        file_kind=file_kind,
        embedded_image_columns=embedded_image_columns,
    )
    path = Path(image_source_path) if image_source_path else None

    if mode == "excel_embedded" and path and path.is_file():
        excel_row = row_index + 2
        col_idx = columns.index(primary) if primary in columns else -1
        target_row = excel_row - 1
        if excel_image_index is not None and col_idx >= 0:
            got = excel_image_index.get((target_row, col_idx))
        else:
            got = extract_image_from_excel_cell(
                path,
                excel_row=excel_row,
                column_name=primary,
                columns=columns,
            )
        if got:
            data, mime = got
            return [{"bytes": data, "mime": mime}]

    if mode in ("image_url", "excel_embedded"):
        val = row_dict.get(primary, "")
        urls = extract_image_urls_from_value(val)
        out: List[Dict[str, Any]] = []
        for url in urls[:3]:
            try:
                data, mime = fetch_image_bytes(url)
                out.append({"bytes": data, "mime": mime})
            except Exception:
                continue
        return out

    return []


def vision_prompt_suffix() -> str:
    return "\n\n【学生作答】见上图（图片），请结合题干、标答与评分标准判断。"
