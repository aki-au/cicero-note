
from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Text + JSON helpers

BASE_DIR = "your-base-dir"

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
INDEX_DIR = os.path.join(DATA_DIR,"medlineplus_index" )
def clean_markdown(text: str) -> str:
    if text is None:
        return ""
    content = str(text)
    content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    content = re.sub(r"\*(.*?)\*", r"\1", content)
    content = re.sub(r"#{1,6}\s*", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def clamp_to_1_sentence(text: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    sents = [s for s in sents if s]
    return (sents[0] if sents else (text or "")).strip()


def safe_json_extract(raw: str) -> Optional[Any]:
    if raw is None:
        return None
    txt = str(raw).strip()
    txt = txt.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(txt)
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", txt, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    return None


def load_json_file(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

# Vitals parsing + abnormal detection
# (refactored from visualization.py you uploaded)

def _norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_bp(value: Any) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    s = str(value).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    # If it's BP, do not treat it as float
    if _parse_bp(s):
        return None

    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def build_normal_ranges(normal_ranges_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build normalized lookup for vitals ranges from normal_ranges.json structure:
    {
      "vitals": [
        {"name": "...", "unit": "...", "label": "...", "normal_range": {"min": x, "max": y}, "aliases": [...]},
        ...
      ]
    }
    """
    vitals = (normal_ranges_data or {}).get("vitals", []) or []
    out: Dict[str, Dict[str, Any]] = {}

    def _safe_get(d, keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    for v in vitals:
        if not isinstance(v, dict):
            continue

        name = v.get("name") or v.get("vital") or v.get("key")
        if not name:
            continue

        label = v.get("label") or name
        unit = v.get("unit") or _safe_get(v, ["normal_range", "unit"]) or ""

        low = _to_float(_safe_get(v, ["normal_range", "min"]))
        high = _to_float(_safe_get(v, ["normal_range", "max"]))

        if low is None or high is None:
            continue

        base = {"name": name, "label": label, "unit": unit, "min": low, "max": high}
        out[_norm_key(name)] = base

        for a in (v.get("aliases", []) or []):
            if a:
                out[_norm_key(a)] = base

    return out


def match_vital(vital_name: str, normal_ranges: Dict[str, Dict[str, Any]]) -> Optional[str]:
    vn = _norm_key(vital_name)
    if vn in normal_ranges:
        return vn

    # Substring match
    for k in normal_ranges.keys():
        if vn in k or k in vn:
            return k

    # Token overlap match
    vtok = set(vn.split())
    best_k = None
    best_overlap = 0
    for k in normal_ranges.keys():
        ktok = set(k.split())
        overlap = len(vtok & ktok)
        if overlap > best_overlap:
            best_overlap = overlap
            best_k = k

    if best_k and best_overlap >= 2:
        return best_k

    return None


def format_range(low: float, high: float, unit: str) -> str:
    unit = (unit or "").strip()
    if unit:
        return f"{low}-{high} {unit}"
    return f"{low}-{high}"


def check_abnormal_vitals(vitals: Dict[str, Any], normal_ranges: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

    abnormal: Dict[str, Dict[str, Any]] = {}

    if not isinstance(vitals, dict):
        return abnormal

    for vital_name, reading in vitals.items():
        if reading is None:
            continue

        if isinstance(reading, dict):
            value = reading.get("value", reading.get("val", reading.get("reading")))
            unit = reading.get("unit", reading.get("units"))
        else:
            value = reading
            unit = None

        bp = _parse_bp(value)
        if bp:
            matched = match_vital(vital_name, normal_ranges)
            if not matched:
                continue
            ref = normal_ranges[matched]
            low = ref["min"]
            high = ref["max"]
            sys_val, dia_val = bp
            sys_abn = not (low <= sys_val <= high)
            dia_abn = not (low <= dia_val <= high)
            if sys_abn or dia_abn:
                abnormal[vital_name] = {
                    "value": {"systolic": sys_val, "diastolic": dia_val},
                    "unit": unit or ref.get("unit", ""),
                    "label": ref.get("label", ref.get("name", matched)),
                    "normal_range": format_range(low, high, ref.get("unit", "")),
                }
            continue

        value_f = _to_float(value)
        if value_f is None:
            continue

        matched = match_vital(vital_name, normal_ranges)
        if not matched:
            continue

        ref = normal_ranges[matched]
        low = ref["min"]
        high = ref["max"]

        if not (low <= value_f <= high):
            abnormal[vital_name] = {
                "value": value_f,
                "unit": unit or ref.get("unit", ""),
                "label": ref.get("label", ref.get("name", matched)),
                "normal_range": format_range(low, high, ref.get("unit", "")),
            }

    return abnormal

# Markdown -> PDF (ReportLab)

def markdown_to_pdf_bytes(markdown_text: str, title: str = "Clinical Note Report") -> bytes:

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    try:
        # Common path in many linux envs; harmless if missing
        dejavu_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(dejavu_path):
            pdfmetrics.registerFont(TTFont("DejaVu", dejavu_path))
            body_font = "DejaVu"
        else:
            body_font = "Helvetica"
    except Exception:
        body_font = "Helvetica"

    x_margin = 0.75 * inch
    y = height - 0.9 * inch

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 0.9 * inch

    def draw_line(text: str, font: str, size: int, indent: float = 0.0, leading: float = 1.25):
        nonlocal y
        c.setFont(font, size)

        max_width = width - x_margin * 2 - indent
       
        approx_chars = max(40, int(max_width / (size * 0.55)))

        words = (text or "").split()
        if not words:
            y -= size * leading * 0.6
            return

        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if len(test) <= approx_chars:
                line = test
            else:
                if y < 1.0 * inch:
                    new_page()
                c.drawString(x_margin + indent, y, line)
                y -= size * leading
                line = w

        if line:
            if y < 1.0 * inch:
                new_page()
            c.drawString(x_margin + indent, y, line)
            y -= size * leading

   
    draw_line(title, "Helvetica-Bold", 18)
    y -= 8

    md = (markdown_text or "").strip().splitlines()
    for raw in md:
        line = raw.rstrip()

        if not line.strip():
            y -= 8
            continue

        if line.startswith("### "):
            draw_line(line[4:].strip(), "Helvetica-Bold", 12)
            y -= 2
            continue

        if line.startswith("## "):
            draw_line(line[3:].strip(), "Helvetica-Bold", 14)
            y -= 4
            continue

        if line.startswith("# "):
            draw_line(line[2:].strip(), "Helvetica-Bold", 16)
            y -= 6
            continue

        bullet = None
        if re.match(r"^\s*[-*]\s+", line):
            bullet = "•"
            content = re.sub(r"^\s*[-*]\s+", "", line).strip()
            draw_line(f"{bullet} {content}", body_font, 11, indent=12)
            continue

        
        draw_line(line.strip(), body_font, 11)

        if y < 1.0 * inch:
            new_page()

    c.save()
    buf.seek(0)
    return buf.read()
