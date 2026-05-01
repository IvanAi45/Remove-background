"""
Material composition extraction for clothing labels, with optional OCR.

This module is designed to be standalone and runnable without touching the
existing project server (`app.py`). If OCR isn't available, you can still use
`parse_materials_from_text()` (or the CLI) to test the parsing pipeline.
"""

from __future__ import annotations

import re
import io
import unicodedata
import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path


@dataclass
class MaterialItem:
    """A single material composition entry."""

    key: str
    name_en: str
    name_zh: str
    percent: float
    icon: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name_en": self.name_en,
            "name_zh": self.name_zh,
            "percent": self.percent,
            "icon": self.icon,
            # UI contract: always return English display label.
            "label": f"{self.name_en} {int(self.percent)}%",
        }


@dataclass
class LabelAnalysisResult:
    """Analysis result for one label image or a plain text snippet."""

    materials: list[MaterialItem] = field(default_factory=list)
    raw_text: str = ""
    ocr_engine: str | None = None
    ocr_error: str | None = None
    translation_applied: bool = False
    processed_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "materials": [m.to_dict() for m in self.materials],
            "raw_text": self.raw_text,
            "ocr_engine": self.ocr_engine,
            "ocr_error": self.ocr_error,
            "translation_applied": self.translation_applied,
            "processed_text": self.processed_text,
            "notes": self.notes,
        }


# Canonical key -> (name_en, name_zh, icon id for frontend)
# Note: to avoid non-ASCII in source (some linters/encodings), Chinese strings
# are expressed via unicode escapes.
_FABRIC_DEFS: dict[str, tuple[str, str, str]] = {
    "cotton": ("Cotton", "\u68c9", "cotton"),
    "polyester": ("Polyester", "\u805a\u916f\u7ea4\u7ef4", "polyester"),
    "wool": ("Wool", "\u7f8a\u6bdb", "wool"),
    "silk": ("Silk", "\u4e1d", "silk"),
    "linen": ("Linen", "\u4e9a\u9ebb", "linen"),
    "viscose": ("Viscose", "\u7c98\u80f6\u7ea4\u7ef4", "viscose"),
    "rayon": ("Rayon", "\u4eba\u9020\u4e1d", "viscose"),
    "nylon": ("Nylon", "\u9526\u7eb6", "nylon"),
    "spandex": ("Spandex", "\u6c28\u7eb6", "spandex"),
    "elastane": ("Elastane", "\u5f39\u6027\u7ea4\u7ef4", "spandex"),
    "lycra": ("Lycra", "\u83b1\u5361", "spandex"),
    "acrylic": ("Acrylic", "\u8148\u7eb6", "acrylic"),
    "cashmere": ("Cashmere", "\u7f8a\u7ed2", "wool"),
    "modal": ("Modal", "\u83ab\u4ee3\u5c14", "viscose"),
    "hemp": ("Hemp", "\u9ebb", "linen"),
    "bamboo": ("Bamboo fiber", "\u7af9\u7ea4\u7ef4", "cotton"),
    "acetate": ("Acetate", "\u918b\u916f\u7ea4\u7ef4", "viscose"),
    "polyurethane": ("Polyurethane", "\u805a\u6c28\u916f", "nylon"),
    "down": ("Down", "\u7fbd\u7ed2", "down"),
    "leather": ("Leather", "\u76ae\u9769", "leather"),
    "fur": ("Fur", "\u6bdb\u76ae", "wool"),
    "angora": ("Angora", "\u5b89\u54e5\u62c9\u5154\u6bdb", "wool"),
    "mohair": ("Mohair", "\u9a6c\u6d77\u6bdb", "wool"),
    "alpaca": ("Alpaca", "\u7f8a\u9a7c\u6bdb", "wool"),
    "metal": ("Metal fiber", "\u91d1\u5c5e\u7ea4\u7ef4", "nylon"),
}

# Lowercased alias -> canonical key
_ALIASES: dict[str, str] = {}
for _k, (_en, _zh, _) in _FABRIC_DEFS.items():
    _ALIASES[_k] = _k
    _ALIASES[_en.lower()] = _k
    _ALIASES[_zh.lower()] = _k

_extra_aliases = {
    "\u68c9": "cotton",
    "\u7eaf\u68c9": "cotton",
    "\u5168\u68c9": "cotton",
    "\u6da4\u7eb6": "polyester",
    "\u805a\u916f": "polyester",
    "\u7f8a\u6bdb": "wool",
    "\u771f\u4e1d": "silk",
    "\u6851\u8695\u4e1d": "silk",
    "\u4e9a\u9ebb": "linen",
    "\u82ce\u9ebb": "linen",
    "\u7c98\u7ea4": "viscose",
    "\u7c98\u80f6": "viscose",
    "\u9526\u7eb6": "nylon",
    "\u5c3c\u9f99": "nylon",
    "\u6c28\u7eb6": "spandex",
    "\u83b1\u5361": "lycra",
    "\u8148\u7eb6": "acrylic",
    "\u5f00\u53f8\u7c73": "cashmere",
    "\u7f8a\u7ed2": "cashmere",
    "modal": "modal",
    "\u83ab\u4ee3\u5c14": "modal",
    "\u7af9\u7ea4\u7ef4": "bamboo",
    "\u805a\u916f\u7ea4\u7ef4": "polyester",
    "\u7fbd\u7ed2": "down",
    "\u7070\u9e2d\u7ed2": "down",
    "\u767d\u9e2d\u7ed2": "down",
    "pu": "polyurethane",
    "\u771f\u76ae": "leather",
    "\u725b\u76ae": "leather",
    "\u7f8a\u76ae": "leather",
    # Spanish/Italian/French-like aliases commonly found on labels.
    "lana": "wool",
    "viscosa": "viscose",
    "poliester": "polyester",
    "poliéster": "polyester",
    "acrilico": "acrylic",
    "acrílico": "acrylic",
    "algodon": "cotton",
    "algodón": "cotton",
    "cuero": "leather",
    "pluma": "down",
}
for _a, _k in _extra_aliases.items():
    _ALIASES[_a.lower()] = _k


def _normalize_ocr_text(text: str) -> str:
    """Normalize whitespace and common OCR quirks."""
    t = text.replace("\u3000", " ").replace("\uff05", "%").replace("°", "%").replace("º", "%")
    # OCR often confuses O/I/l with 0/1 in percentage numbers.
    def _fix_pct_digits(match: re.Match[str]) -> str:
        token = match.group(1)
        token = token.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
        return f"{token}%"
    t = re.sub(r"([0-9OoilIl]{1,3})\s*%", _fix_pct_digits, t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _resolve_fabric_token(token: str) -> str | None:
    """Map an OCR/token fragment to a canonical fabric key."""
    # Strip a small set of punctuation; other punctuation is normalized earlier.
    raw = token.strip().strip(".,;:")
    if not raw:
        return None
    key = raw.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Latin fold handles accents, e.g. "acrílico" -> "acrilico".
    key_folded = unicodedata.normalize("NFKD", key).encode("ascii", "ignore").decode("ascii")
    if key_folded in _ALIASES:
        return _ALIASES[key_folded]
    # Lightweight fuzzy matching for English stems / OCR noise.
    for alias, canon in _ALIASES.items():
        if len(alias) >= 3 and (
            key == alias
            or key.startswith(alias)
            or alias in key
            or key_folded == alias
            or key_folded.startswith(alias)
            or alias in key_folded
        ):
            if len(key_folded) <= 2 and len(alias) > 4:
                continue
            return canon
    return None


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def translate_zh_to_en(text: str) -> str:
    """
    Translate common Chinese composition tokens into English.

    This is intentionally deterministic and offline: we translate *materials* and
    composition patterns, not the entire label content. The goal is to feed an
    English-like string into the downstream parser.
    """
    if not text:
        return ""

    t = _normalize_ocr_text(text)

    # Replace a few common full-width/Chinese punctuation forms.
    t = (
        t.replace("\uff1a", ":")
        .replace("\uff0c", " ")
        .replace("\u3002", " ")
        .replace("\uff1b", " ")
        .replace("\u3001", " ")
    )

    # Translate material words first (longer keys first to avoid partial overlaps).
    # The keys are unicode-escaped to keep the source ASCII-only.
    zh_to_en = {
        "\u805a\u916f\u7ea4\u7ef4": "Polyester",
        "\u805a\u916f": "Polyester",
        "\u6da4\u7eb6": "Polyester",
        "\u7c98\u80f6\u7ea4\u7ef4": "Viscose",
        "\u7c98\u80f6": "Viscose",
        "\u7c98\u7ea4": "Viscose",
        "\u4eba\u9020\u4e1d": "Rayon",
        "\u9526\u7eb6": "Nylon",
        "\u5c3c\u9f99": "Nylon",
        "\u6c28\u7eb6": "Spandex",
        "\u5f39\u6027\u7ea4\u7ef4": "Elastane",
        "\u83b1\u5361": "Lycra",
        "\u8150\u7eb6": "Acrylic",
        "\u8148\u7eb6": "Acrylic",
        "\u7f8a\u7ed2": "Cashmere",
        "\u5f00\u53f8\u7c73": "Cashmere",
        "\u7f8a\u6bdb": "Wool",
        "\u771f\u4e1d": "Silk",
        "\u6851\u8695\u4e1d": "Silk",
        "\u4e9a\u9ebb": "Linen",
        "\u82ce\u9ebb": "Linen",
        "\u9ebb": "Hemp",
        "\u7af9\u7ea4\u7ef4": "Bamboo fiber",
        "\u83ab\u4ee3\u5c14": "Modal",
        "\u918b\u916f\u7ea4\u7ef4": "Acetate",
        "\u805a\u6c28\u916f": "Polyurethane",
        "\u7fbd\u7ed2": "Down",
        "\u767d\u9e2d\u7ed2": "Down",
        "\u7070\u9e2d\u7ed2": "Down",
        "\u76ae\u9769": "Leather",
        "\u771f\u76ae": "Leather",
        "\u725b\u76ae": "Leather",
        "\u7f8a\u76ae": "Leather",
        "\u68c9": "Cotton",
        "\u7eaf\u68c9": "Cotton",
        "\u5168\u68c9": "Cotton",
    }
    for zh in sorted(zh_to_en.keys(), key=len, reverse=True):
        t = t.replace(zh, zh_to_en[zh])

    # Normalize patterns like: "Cotton 65%" -> "65% Cotton" (English-friendly).
    # Avoid `\b` after `%` because `%` isn't a word character.
    t = re.sub(r"(?i)\b([A-Za-z][A-Za-z \-]{1,28}?)\s+(\d{1,3})\s*%", r"\2% \1", t)

    return t


def translate_to_english_if_needed(text: str) -> str:
    """Translate CJK-heavy text to an English-like representation before parsing."""
    if _contains_cjk(text):
        return translate_zh_to_en(text)
    return text or ""


def prepare_text_for_parsing(text: str) -> tuple[str, bool]:
    """Return (processed_text, translation_applied)."""
    processed = translate_to_english_if_needed(text or "")
    return processed, (processed != (text or ""))


def _extract_percent_material_pairs(text: str) -> list[tuple[float, str]]:
    """Extract (percent, material-token) pairs from a string."""
    t = _normalize_ocr_text(text)
    pairs: list[tuple[float, str]] = []

    # 95% cotton (English materials after %).
    # OCR noise sometimes inserts 1-3 garbage chars/digits between '%' and token,
    # e.g. "100% 2 POLYESTER", so we tolerate a short non-letter segment.
    for m in re.finditer(
        r"(\d{1,3})\s*%\s*(?:[^A-Za-z\u4e00-\u9fff]{0,4})\s*([A-Za-z][A-Za-z\s\-]{0,28}?)(?=\s*\d|\s*%|[^\w\-]|$)",
        t,
        re.I,
    ):
        p = float(m.group(1))
        if p > 100:
            continue
        word = m.group(2).strip()
        pairs.append((p, word))

    # CJK immediately after % (kept for compatibility; translation-first is preferred).
    # the preferred pipeline translates to English first.
    for m in re.finditer(
        r"(\d{1,3})\s*%([\u4e00-\u9fff]{1,8})",
        t,
    ):
        p = float(m.group(1))
        if p > 100:
            continue
        word = m.group(2).strip()
        pairs.append((p, word))

    # material before percent (English or CJK).
    for m in re.finditer(
        r"([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff\s\-]{1,24}?)\s*[:\uff1a]?\s*(\d{1,3})\s*%",
        t,
        re.I,
    ):
        word = m.group(1).strip()
        p = float(m.group(2))
        if p > 100:
            continue
        pairs.append((p, word))

    return pairs


def parse_materials_from_text(text: str) -> list[MaterialItem]:
    """
    Parse materials & percentages from OCR text or manually provided text.

    If the input is primarily Chinese, we first translate the material tokens
    into an English-like representation, then continue the normal parsing flow.
    """
    if not text or not text.strip():
        return []

    processed_text, _ = prepare_text_for_parsing(text)
    pairs = _extract_percent_material_pairs(processed_text)
    by_key_percent: dict[tuple[str, int], MaterialItem] = {}

    for percent, word in pairs:
        # A token may contain multiple words, e.g. "polyester cotton blend".
        # We pick the first resolvable fragment.
        parts = re.split(r"[/+\s]+", word)
        resolved = None
        for part in parts:
            rk = _resolve_fabric_token(part)
            if rk:
                resolved = rk
                break
        if not resolved:
            resolved = _resolve_fabric_token(word)
        if not resolved:
            continue

        en, zh, icon = _FABRIC_DEFS[resolved]
        # Keep different percentages for the same material key so nested sections
        # like shell/lining are not collapsed into a single value.
        k = (resolved, int(round(percent)))
        if k not in by_key_percent:
            by_key_percent[k] = MaterialItem(
                key=resolved, name_en=en, name_zh=zh, percent=percent, icon=icon
            )

    out = sorted(by_key_percent.values(), key=lambda x: -x.percent)
    return out


def _preprocess_for_ocr(img: Image.Image, max_side: int = 2000) -> Image.Image:
    """Basic OCR preprocessing (resize + contrast) for label photos."""
    im = img.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if max(w, h) < 800:
        scale = min(2.5, max_side / max(w, h))
    if scale != 1.0:
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(im)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.35)
    return gray


def _score_ocr_candidate(text: str) -> int:
    """
    Score OCR output quality for material parsing.

    Higher is better:
    - percentage tokens
    - known material keywords/aliases
    - parsed material pairs
    """
    if not text or not text.strip():
        return 0
    t = _normalize_ocr_text(text)
    pct_hits = len(re.findall(r"\d{1,3}\s*%", t))
    token_hits = 0
    lower = t.lower()
    for alias in _ALIASES.keys():
        if len(alias) < 3:
            continue
        if alias in lower:
            token_hits += 1
    try:
        parsed_count = len(parse_materials_from_text(t))
    except Exception:
        parsed_count = 0
    # parsed_count weighted heavily because it is the end goal.
    return pct_hits * 2 + token_hits + parsed_count * 6


def _count_parsed_pairs(text: str) -> int:
    """Count parsed material pairs for OCR early stopping."""
    try:
        return len(parse_materials_from_text(text))
    except Exception:
        return 0


def ocr_label_image_bytes(data: bytes) -> tuple[str, str | None]:
    """
    OCR an image byte buffer and return (text, error_message).

    Uses the system `tesseract` CLI directly to avoid Python package ABI issues.
    """
    tesseract_cmd = _resolve_tesseract_executable()
    if tesseract_cmd is None:
        return (
            "",
            "Missing system OCR engine: tesseract executable not found in PATH. "
            "Install Tesseract-OCR, OR set env `TESSERACT_CMD` to full exe path.",
        )

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        return "", f"Failed to read image: {e}"

    proc = _preprocess_for_ocr(img)
    text = ""
    last_err: str | None = None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        variants: list[Image.Image] = [
            proc,
            proc.rotate(90, expand=True),
            proc.rotate(270, expand=True),
            proc.rotate(180, expand=True),
        ]
        best_score = -1
        best_text = ""
        candidates: list[tuple[int, str]] = []

        def run_stage(
            use_variants: list[Image.Image],
            langs: tuple[str, ...],
            psms: tuple[str, ...],
            min_pairs_to_stop: int,
        ) -> bool:
            nonlocal best_score, best_text, last_err, candidates
            for v in use_variants:
                v.save(tmp_path, format="PNG")
                for lang in langs:
                    for psm in psms:
                        cmd = [tesseract_cmd, tmp_path, "stdout", "-l", lang, "--psm", psm]
                        try:
                            run = subprocess.run(
                                cmd,
                                check=False,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="ignore",
                            )
                        except Exception as e:
                            last_err = str(e)
                            continue

                        out = (run.stdout or "").strip()
                        if run.returncode != 0:
                            stderr = (run.stderr or "").strip()
                            if stderr:
                                last_err = stderr
                            continue
                        if not out:
                            continue
                        score = _score_ocr_candidate(out)
                        candidates.append((score, out))
                        if score > best_score:
                            best_score = score
                            best_text = out
                        if _count_parsed_pairs(out) >= min_pairs_to_stop:
                            return True
            return False

        # Fast stage first (significantly reduces average latency).
        found_enough = run_stage(
            use_variants=variants[:3],
            langs=("eng+spa", "eng"),
            psms=("6",),
            min_pairs_to_stop=2,
        )
        # Wider fallback only when fast stage is insufficient.
        if not found_enough:
            run_stage(
                use_variants=variants,
                langs=("eng+spa+chi_sim", "chi_sim+eng", "eng+spa", "eng"),
                psms=("6", "4", "11"),
                min_pairs_to_stop=1,
            )

        text = best_text
        # If the best candidate still has too few pairs, merge top candidates to
        # recover missing percentages from different OCR variants.
        if _count_parsed_pairs(text) < 2 and candidates:
            seen: set[str] = set()
            merged: list[str] = []
            for _, out in sorted(candidates, key=lambda x: x[0], reverse=True):
                if out in seen:
                    continue
                seen.add(out)
                merged.append(out)
                if len(merged) >= 4:
                    break
            if merged:
                text = "\n".join(merged)

        if not text.strip() and last_err:
            return "", f"Tesseract OCR failed: {last_err}"
        return text, None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _resolve_tesseract_executable() -> str | None:
    """
    Resolve the tesseract executable path with several fallbacks:
    1) `TESSERACT_CMD` environment variable
    2) PATH lookup
    3) Common Windows installation locations
    """
    env_cmd = (os.environ.get("TESSERACT_CMD") or "").strip().strip('"')
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    from_path = shutil.which("tesseract")
    if from_path:
        return from_path

    # Common Windows install paths.
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def analyze_label_image_bytes(data: bytes) -> LabelAnalysisResult:
    """
    Full pipeline: OCR (if available) + translation (if needed) + parsing.
    """
    notes: list[str] = []
    raw, ocr_err = ocr_label_image_bytes(data)
    engine = "tesseract_cli" if raw else None
    if ocr_err and not raw:
        notes.append(ocr_err)
    processed_text, translation_applied = prepare_text_for_parsing(raw)
    materials = parse_materials_from_text(raw)
    if raw and not materials:
        notes.append(
            "No composition pairs found. Try a closer photo of the composition lines, "
            "or provide the label text manually."
        )

    return LabelAnalysisResult(
        materials=materials,
        raw_text=raw,
        ocr_engine=engine,
        ocr_error=ocr_err if not raw else None,
        translation_applied=translation_applied,
        processed_text=processed_text,
        notes=notes,
    )


def analyze_from_text_only(text: str) -> LabelAnalysisResult:
    """Skip OCR and only run translation+parsing (demo/testing)."""
    processed_text, translation_applied = prepare_text_for_parsing(text)
    materials = parse_materials_from_text(text)
    notes: list[str] = []
    if text.strip() and not materials:
        notes.append(
            "No recognizable composition pairs found. Examples: "
            "80% Cotton, Cotton 80%, \u68c9 65%, \u805a\u916f\u7ea4\u7ef4 35%"
        )
    return LabelAnalysisResult(
        materials=materials,
        raw_text=text,
        ocr_engine=None,
        ocr_error=None,
        translation_applied=translation_applied,
        processed_text=processed_text,
        notes=notes,
    )


def _result_to_printable_dict(result: LabelAnalysisResult) -> dict[str, Any]:
    data = result.to_dict()
    data["materials"] = [
        {**m, "percent": float(m["percent"])} for m in (data.get("materials") or [])
    ]
    return data


def _cli() -> int:
    """
    CLI entrypoint.

    Examples:
      python material_recognition.py --text \"\u68c9 65% \u805a\u916f\u7ea4\u7ef4 35%\"\n+      python material_recognition.py --text \"65% Cotton 35% Polyester\"\n+      python material_recognition.py --image path/to/label.jpg\n+    """
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Material composition recognition (standalone).")
    parser.add_argument("--text", type=str, default="", help="Label text to parse.")
    parser.add_argument("--image", type=str, default="", help="Path to a label image for OCR.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local demo server that serves frontend/label_demo.html and API endpoints.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Demo server host.")
    parser.add_argument("--port", type=int, default=5055, help="Demo server port.")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not auto-open the browser when starting the demo server.",
    )
    args = parser.parse_args()

    if args.serve or (not args.text and not args.image):
        return _serve_demo(host=args.host, port=int(args.port), open_browser=(not args.no_open))

    if args.image:
        try:
            data = Path(args.image).read_bytes()
        except Exception as e:
            print(json.dumps({"error": f"Failed to read image: {e}"}, ensure_ascii=False, indent=2))
            return 2
        result = analyze_label_image_bytes(data)
        print(json.dumps(_result_to_printable_dict(result), ensure_ascii=False, indent=2))
        return 0

    result = analyze_from_text_only(args.text or "")
    print(json.dumps(_result_to_printable_dict(result), ensure_ascii=False, indent=2))
    return 0


def _serve_demo(host: str = "127.0.0.1", port: int = 5055, open_browser: bool = True) -> int:
    """
    Serve the demo page and provide API endpoints for end-to-end testing.

    - GET /              -> serves `frontend/label_demo.html`
    - GET /api/health    -> health check
    - POST /api/recognize   (multipart: file) -> OCR + parse materials
    - POST /api/parse-text  (json: {text})    -> translate+parse only
    """
    try:
        from flask import Flask, jsonify, request, send_from_directory
        from flask_cors import CORS
    except Exception as e:
        print(
            "Demo server requires Flask. Install: pip install flask flask-cors\n"
            f"Import error: {e}"
        )
        return 2

    backend_dir = Path(__file__).resolve().parent
    frontend_dir = backend_dir.parent / "frontend"
    demo_file = "label_demo.html"
    if not (frontend_dir / demo_file).exists():
        print(f"Missing demo page: {frontend_dir / demo_file}")
        return 2

    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

    @app.get("/")
    def _index():
        return send_from_directory(str(frontend_dir), demo_file)

    @app.get("/api/health")
    def _health():
        return jsonify({"ok": True, "service": "material-recognition-demo"})

    @app.post("/api/parse-text")
    def _parse_text():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text") or ""
        result = analyze_from_text_only(str(text))
        translated = translate_to_english_if_needed(str(text))
        data = result.to_dict()
        data["translated_text"] = translated
        return jsonify(data)

    @app.post("/api/recognize")
    def _recognize():
        if "file" not in request.files:
            return jsonify({"error": "Missing form field: file"}), 400
        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "No file selected"}), 400

        data = f.read()
        result = analyze_label_image_bytes(data)
        # Provide a translated version to help debugging.
        translated = translate_to_english_if_needed(result.raw_text)
        out = result.to_dict()
        out["translated_text"] = translated
        return jsonify(out)

    url = f"http://{host}:{port}/"
    print(f"Demo server running at: {url}")

    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    app.run(host=host, port=port, debug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
