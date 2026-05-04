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
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
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


# Keep this map at module level so translation does not rebuild it per call.
_ZH_TO_EN_MAP: dict[str, str] = {
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
_ZH_TO_EN_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_ZH_TO_EN_MAP.keys(), key=len, reverse=True))
)


@lru_cache(maxsize=2048)
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

    # One regex pass is faster than dozens of sequential .replace() calls.
    t = _ZH_TO_EN_PATTERN.sub(lambda m: _ZH_TO_EN_MAP[m.group(0)], t)

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


def _parse_materials_from_processed_text(processed_text: str) -> list[MaterialItem]:
    """Parse percentages/materials from already prepared English-like text."""
    pairs = _extract_percent_material_pairs(processed_text)
    by_key_percent: dict[tuple[str, int], MaterialItem] = {}

    for percent, word in pairs:
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
        k = (resolved, int(round(percent)))
        if k not in by_key_percent:
            by_key_percent[k] = MaterialItem(
                key=resolved, name_en=en, name_zh=zh, percent=percent, icon=icon
            )
    return sorted(by_key_percent.values(), key=lambda x: -x.percent)


def parse_materials_from_text(text: str) -> list[MaterialItem]:
    """
    Parse materials & percentages from OCR text or manually provided text.

    If the input is primarily Chinese, we first translate the material tokens
    into an English-like representation, then continue the normal parsing flow.
    """
    if not text or not text.strip():
        return []

    processed_text, _ = prepare_text_for_parsing(text)
    return _parse_materials_from_processed_text(processed_text)


def _resize_for_ocr_rgb(img: Image.Image, max_side: int = 2000) -> Image.Image:
    """Resize RGB so small labels are upscaled and huge photos are downscaled."""
    im = img.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if max(w, h) < 800:
        scale = min(2.5, max_side / max(w, h))
    if scale != 1.0:
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return im


def _gray_standard(rgb: Image.Image) -> Image.Image:
    """Baseline grayscale pipeline (good for scans and clean photos)."""
    gray = ImageOps.grayscale(rgb)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    return ImageEnhance.Contrast(gray).enhance(1.35)


def _gray_photo_median(rgb: Image.Image) -> Image.Image:
    """
    Grayscale tuned for real photos: thin fabric show-through, glare, moire.

    Median filter suppresses high-frequency bleed-through and pepper noise.
    """
    gray = ImageOps.grayscale(rgb)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageOps.autocontrast(gray, cutoff=3)
    return ImageEnhance.Contrast(gray).enhance(1.45)


def _gray_otsu_binarize(gray: Image.Image) -> Image.Image | None:
    """Global Otsu binarization (helps when background is uneven)."""
    try:
        import numpy as np
    except ImportError:
        return None
    arr = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    p = hist / max(arr.size, 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b))
    bin_arr = np.where(arr > t, 255, 0).astype(np.uint8)
    return Image.fromarray(bin_arr, mode="L")


def _gray_opencv_adaptive(rgb: Image.Image) -> Image.Image | None:
    """Optional OpenCV adaptive threshold (strong on uneven lighting)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    bgr = cv2.cvtColor(np.asarray(rgb.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, d=7, sigmaColor=55, sigmaSpace=55)
    ath = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 8
    )
    return Image.fromarray(ath, mode="L")


def _scanify_with_opencv(rgb: Image.Image) -> Image.Image | None:
    """
    Try document-style perspective correction (scan conversion).

    This is most useful for real photos with perspective skew. For already-flat
    scans it usually returns None quickly and we fall back to the original image.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    arr = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if min(h, w) < 240:
        return None

    # Work on a smaller proxy for speed.
    max_dim = 900
    scale = max(h, w) / max_dim if max(h, w) > max_dim else 1.0
    sh, sw = int(h / scale), int(w / scale)
    small = cv2.resize(arr, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 50, 150)
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    def area(c):
        return float(cv2.contourArea(c))

    target = None
    img_area = float(sw * sh)
    for c in sorted(contours, key=area, reverse=True)[:12]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        a = area(approx)
        if a < img_area * 0.08:
            continue
        target = approx.reshape(4, 2).astype(np.float32)
        break
    if target is None:
        return None

    # Order points TL, TR, BR, BL.
    pts = target
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype=np.float32) * float(scale)

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    width = int(max(dist(src[0], src[1]), dist(src[2], src[3])))
    height = int(max(dist(src[1], src[2]), dist(src[0], src[3])))
    if width < 80 or height < 80:
        return None

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, m, (width, height), flags=cv2.INTER_LINEAR)
    return Image.fromarray(warped, mode="RGB")


def _build_ocr_gray_variants(img: Image.Image) -> list[Image.Image]:
    """
    Build several grayscale images for Tesseract.

    Order matters for speed: simpler variants first; heavy / binary variants
    are used when the fast OCR pass is weak.
    """
    rgb = _resize_for_ocr_rgb(img)
    variants: list[Image.Image] = []
    rgb_candidates: list[Image.Image] = []

    # 1) original frame
    rgb_candidates.append(rgb)
    # 2) scan-converted frame (if detected)
    scan_rgb = _scanify_with_opencv(rgb)
    if scan_rgb is not None:
        rgb_candidates.append(scan_rgb)

    for rgb_item in rgb_candidates:
        g0 = _gray_standard(rgb_item)
        variants.append(g0)

        g1 = _gray_photo_median(rgb_item)
        if g1.tobytes() != g0.tobytes():
            variants.append(g1)

        otsu = _gray_otsu_binarize(g0)
        if otsu is not None:
            variants.append(otsu)

        cv_bin = _gray_opencv_adaptive(rgb_item)
        if cv_bin is not None:
            variants.append(cv_bin)

    return variants


def _rotate_variants(gray: Image.Image, angles: tuple[int, ...]) -> list[Image.Image]:
    """Return rotated copies (0 first) for vertical / skewed label text."""
    out: list[Image.Image] = []
    for ang in angles:
        if ang == 0:
            out.append(gray)
        else:
            out.append(gray.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC))
    return out


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
        parsed_count = len(_parse_materials_from_processed_text(_normalize_ocr_text(t)))
    except Exception:
        parsed_count = 0
    # parsed_count weighted heavily because it is the end goal.
    return pct_hits * 2 + token_hits + parsed_count * 6


def _count_parsed_pairs(text: str) -> int:
    """Count parsed material pairs for OCR early stopping."""
    try:
        return len(_parse_materials_from_processed_text(_normalize_ocr_text(text)))
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

    gray_bases = _build_ocr_gray_variants(img)
    text = ""
    last_err: str | None = None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        best_score = -1
        best_text = ""
        candidates: list[tuple[int, str]] = []

        def run_stage(
            bases: list[Image.Image],
            angles: tuple[int, ...],
            langs: tuple[str, ...],
            psms: tuple[str, ...],
            min_pairs_to_stop: int,
        ) -> bool:
            nonlocal best_score, best_text, last_err, candidates
            for g in bases:
                for v in _rotate_variants(g, angles):
                    v.save(tmp_path, format="PNG")
                    for lang in langs:
                        for psm in psms:
                            cmd = [
                                tesseract_cmd,
                                tmp_path,
                                "stdout",
                                "-l",
                                lang,
                                "--oem",
                                "1",
                                "--psm",
                                psm,
                            ]
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

        # Fast stage: first 1-2 gray pipelines, common rotations, few Tesseract modes.
        fast_bases = gray_bases[:2] if len(gray_bases) >= 2 else gray_bases[:1]
        found_enough = run_stage(
            bases=fast_bases,
            angles=(0, 90, 270),
            langs=("eng+spa", "eng"),
            psms=("6",),
            min_pairs_to_stop=2,
        )
        # Wider fallback: all gray variants (including binarization), 180deg, more PSM/lang.
        if not found_enough:
            run_stage(
                bases=gray_bases,
                angles=(0, 90, 270, 180),
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
    materials = _parse_materials_from_processed_text(processed_text)
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
    materials = _parse_materials_from_processed_text(processed_text)
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
