"""
stock_engine.py — 스톡사진(Adobe Stock / Shutterstock 등) 심사 기준 특화 엔진

두 가지 일을 합니다.
  1) Pre-flight 진단 : 해상도 / 노출(클리핑) / 선명도 / 노이즈 / 색공간 / 파일 규격을
     로컬에서 정량 측정하고 심사 통과 가능성을 점수로 환산합니다. (API 키 불필요)
  2) 보정 엔진 : 과보정(과채도·할로·클리핑)을 피하는 스톡 전용 프리셋 5종과
     세부 미세조정 파라미터를 Pillow/NumPy 로 적용합니다. (API 키 불필요)

Streamlit 에 의존하지 않는 순수 모듈이므로 self_check.py 에서 그대로 테스트할 수 있습니다.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

# ---------------------------------------------------------------------------
# 스톡사진 기준 상수
# ---------------------------------------------------------------------------
ADOBE_MIN_MEGAPIXELS = 4.0        # Adobe Stock 사진 최소 4 MP
ADOBE_RECOMMENDED_MEGAPIXELS = 6.0
ADOBE_MAX_FILE_MB = 100.0         # 업로드 파일 최대 용량
ADOBE_MIN_SIDE = 1600             # 실무상 권장 최소 변 길이

JPEG_QUALITY_MIN = 95             # 스톡 제출용 최소 품질
JPEG_QUALITY_MAX = 98

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# Adobe Stock 공식 카테고리 (1~21)
ADOBE_CATEGORIES: List[Tuple[int, str]] = [
    (1, "Animals"),
    (2, "Buildings and Architecture"),
    (3, "Business"),
    (4, "Drinks"),
    (5, "The Environment"),
    (6, "States of Mind"),
    (7, "Food"),
    (8, "Graphic Resources"),
    (9, "Hobbies and Leisure"),
    (10, "Industry"),
    (11, "Landscapes"),
    (12, "Lifestyle"),
    (13, "People"),
    (14, "Plants and Flowers"),
    (15, "Culture and Religion"),
    (16, "Science"),
    (17, "Social Issues"),
    (18, "Sports"),
    (19, "Technology"),
    (20, "Transport"),
    (21, "Travel"),
]
ADOBE_CATEGORY_MAP = {name.lower(): num for num, name in ADOBE_CATEGORIES}


# ---------------------------------------------------------------------------
# 프리셋 정의
# ---------------------------------------------------------------------------
# 모든 값은 "스톡 안전 범위" 안에서 설계되었습니다.
#  - 과채도 금지: vibrance 위주 + saturation_cap 으로 상한 제한
#  - 할로 방지  : sharpen_threshold 로 강한 엣지에서는 샤프닝을 건너뜀
#  - 클리핑 방지: highlight_protect / shadow_lift 로 여유를 남김
PRESETS: Dict[str, Dict[str, Any]] = {
    "natural_commercial": {
        "label": "Natural Commercial",
        "label_ko": "균형 잡힌 표준 상업 스톡 톤",
        "desc": "가장 무난한 기본값. 화이트밸런스와 노출만 정돈하고 색은 거의 건드리지 않아 "
                "디자이너가 2차 편집하기 좋습니다.",
        "wb_strength": 0.75,
        "exposure": 0.04,
        "highlight_protect": 0.45,
        "shadow_lift": 0.28,
        "contrast": 1.04,
        "vibrance": 0.20,
        "saturation_cap": 0.88,
        "chroma_nr": 0.25,
        "sharpen": 0.30,
        "sharpen_threshold": 0.045,
        "black_point": 0.004,
        "warmth": 0.0,
    },
    "bright_airy": {
        "label": "Bright & Airy",
        "label_ko": "밝고 깨끗한 라이프스타일/인테리어 톤",
        "desc": "하이키 톤. 밝기를 살짝 올리고 암부를 부드럽게 들어 올려 깨끗한 느낌을 냅니다. "
                "흰 벽·침구·카페 소품 사진에 적합합니다.",
        "wb_strength": 0.85,
        "exposure": 0.20,
        "highlight_protect": 0.60,
        "shadow_lift": 0.42,
        "contrast": 1.00,
        "vibrance": 0.26,
        "saturation_cap": 0.86,
        "chroma_nr": 0.20,
        "sharpen": 0.22,
        "sharpen_threshold": 0.05,
        "black_point": 0.000,
        "warmth": 0.006,
    },
    "crisp_architecture": {
        "label": "Crisp Architecture & Product",
        "label_ko": "선명한 디테일·중립 화이트밸런스",
        "desc": "건축·제품·산업 사진용. 화이트밸런스를 중립으로 강하게 잡고 대비와 디테일을 "
                "살리되, 엣지 할로는 임계값으로 억제합니다.",
        "wb_strength": 0.95,
        "exposure": 0.02,
        "highlight_protect": 0.50,
        "shadow_lift": 0.20,
        "contrast": 1.10,
        "vibrance": 0.12,
        "saturation_cap": 0.85,
        "chroma_nr": 0.30,
        "sharpen": 0.55,
        "sharpen_threshold": 0.030,
        "black_point": 0.006,
        "warmth": 0.0,
    },
    "clean_editorial": {
        "label": "Clean Editorial",
        "label_ko": "부드러운 대비·필름 과장 없는 다큐/여행 톤",
        "desc": "다큐·여행·보도성 스톡용. 대비를 낮추고 색은 억제해 사실적이고 차분한 톤을 "
                "만듭니다. 그림 같은 과장이 없습니다.",
        "wb_strength": 0.70,
        "exposure": 0.00,
        "highlight_protect": 0.65,
        "shadow_lift": 0.32,
        "contrast": 0.98,
        "vibrance": 0.14,
        "saturation_cap": 0.82,
        "chroma_nr": 0.25,
        "sharpen": 0.25,
        "sharpen_threshold": 0.05,
        "black_point": 0.002,
        "warmth": 0.0,
    },
    "copy_space_neutral": {
        "label": "Copy Space / Neutral Soft",
        "label_ko": "텍스트 배치용 여백 사진에 어울리는 차분한 톤",
        "desc": "카피 스페이스용. 대비·채도를 낮춰 텍스트를 얹어도 시끄럽지 않게, "
                "밝은 여백이 살아나도록 만듭니다.",
        "wb_strength": 0.80,
        "exposure": 0.10,
        "highlight_protect": 0.50,
        "shadow_lift": 0.45,
        "contrast": 0.94,
        "vibrance": 0.08,
        "saturation_cap": 0.78,
        "chroma_nr": 0.30,
        "sharpen": 0.18,
        "sharpen_threshold": 0.055,
        "black_point": 0.000,
        "warmth": 0.0,
    },
}

PRESET_ORDER = [
    "natural_commercial",
    "bright_airy",
    "crisp_architecture",
    "clean_editorial",
    "copy_space_neutral",
]

# 슬라이더 및 안전 가드 범위
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "wb_strength": (0.0, 1.0),
    "exposure": (-0.60, 0.60),
    "highlight_protect": (0.0, 1.0),
    "shadow_lift": (0.0, 1.0),
    "contrast": (0.85, 1.22),
    "vibrance": (0.0, 0.60),
    "saturation_cap": (0.60, 1.00),
    "chroma_nr": (0.0, 1.0),
    "sharpen": (0.0, 0.80),
    "sharpen_threshold": (0.0, 0.15),
    "black_point": (0.0, 0.02),
    "warmth": (-0.02, 0.02),
}

# 과보정 방지 가드: 스톡 제출용 상한 (사용자가 아무리 올려도 이 선을 넘지 않음)
GUARD_RANGES: Dict[str, Tuple[float, float]] = {
    "wb_strength": (0.0, 1.0),
    "exposure": (-0.35, 0.40),
    "highlight_protect": (0.0, 1.0),
    "shadow_lift": (0.0, 0.65),
    "contrast": (0.90, 1.14),
    "vibrance": (0.0, 0.38),
    "saturation_cap": (0.72, 1.00),
    "chroma_nr": (0.0, 0.80),
    "sharpen": (0.0, 0.62),
    "sharpen_threshold": (0.02, 0.15),
    "black_point": (0.0, 0.012),
    "warmth": (-0.012, 0.012),
}

DEFAULTS: Dict[str, float] = {k: float(PRESETS["natural_commercial"][k]) for k in PARAM_RANGES}


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_params(params: Optional[Dict[str, Any]] = None, guard: bool = True) -> Dict[str, float]:
    """파라미터를 사전 정의된 범위(가드 적용 시 스톡 안전 범위)로 정규화합니다."""
    out: Dict[str, float] = {}
    src = dict(DEFAULTS)
    if params:
        for k, v in params.items():
            if k in PARAM_RANGES and isinstance(v, (int, float)):
                src[k] = float(v)
    for key, (low, high) in PARAM_RANGES.items():
        val = float(src.get(key, DEFAULTS[key]))
        if guard:
            gl, gh = GUARD_RANGES[key]
            val = clamp(val, max(low, gl), min(high, gh))
        else:
            val = clamp(val, low, high)
        out[key] = val
    return out


def preset_params(key: str, guard: bool = True) -> Dict[str, float]:
    return sanitize_params(PRESETS.get(key, PRESETS["natural_commercial"]), guard=guard)


def _gcd_ratio(w: int, h: int) -> str:
    g = math.gcd(int(w), int(h)) or 1
    rw, rh = w // g, h // g
    if rw > 40 or rh > 40:  # 기약분수가 너무 복잡하면 소수비로 표기
        return f"{w / h:.2f}:1"
    return f"{rw}:{rh}"


def _icc_description(icc_bytes: Optional[bytes]) -> Optional[str]:
    if not icc_bytes:
        return None
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        return (ImageCms.getProfileDescription(prof) or "").strip() or None
    except Exception:
        return None


def _is_srgb_ish(desc: Optional[str]) -> bool:
    if not desc:
        return True  # 프로필 없음 = 대부분 sRGB 로 가정
    d = desc.lower()
    return ("srgb" in d) or ("iec61966" in d) or ("apple rgb" in d)


# ---------------------------------------------------------------------------
# 이미지 로드
# ---------------------------------------------------------------------------
def load_image(source: Any) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    경로 / bytes / file-like 어디서든 열어 EXIF 회전을 적용한 RGB 이미지와 메타 정보를 돌려줍니다.
    투명 PNG 는 흰 배경에 합성하고, CMYK/비 sRGB 프로필은 sRGB 로 변환합니다.
    """
    img = Image.open(source)
    img.load()
    fmt = (img.format or "UNKNOWN").upper()
    icc = img.info.get("icc_profile")
    exif_raw = img.info.get("exif")
    icc_desc = _icc_description(icc)

    info: Dict[str, Any] = {
        "format": fmt,
        "source_mode": img.mode,
        "icc_profile": icc,
        "icc_name": icc_desc,
        "is_srgb": _is_srgb_ish(icc_desc),
        "exif_bytes": exif_raw,
        "has_exif": bool(exif_raw),
        "has_icc": bool(icc),
        "converted": [],
    }

    # EXIF 회전
    try:
        rotated = ImageOps.exif_transpose(img)
        if rotated is not None and rotated is not img:
            img = rotated
            info["converted"].append("EXIF 회전 적용")
    except Exception:
        pass

    # 알파 합성 / 색공간 변환
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
        info["converted"].append("투명 영역 흰색 합성")
    elif img.mode == "CMYK":
        converted = False
        if icc:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                dst_prof = ImageCms.createProfile("sRGB")
                img = ImageCms.profileToProfile(img, src_prof, dst_prof, outputMode="RGB")
                converted = True
            except Exception:
                converted = False
        if not converted:
            img = img.convert("RGB")
        info["converted"].append("CMYK → sRGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
        info["converted"].append(f"{info['source_mode']} → RGB")

    return img, info


def convert_to_srgb(img: Image.Image, icc_bytes: Optional[bytes]) -> Image.Image:
    """비 sRGB ICC 프로필을 sRGB 로 변환합니다(가능할 때만)."""
    if not icc_bytes or _is_srgb_ish(_icc_description(icc_bytes)):
        return img
    try:
        src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        dst_prof = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(img, src_prof, dst_prof, outputMode="RGB")
    except Exception:
        return img


def strip_gps(exif_bytes: Optional[bytes]) -> Optional[bytes]:
    """EXIF 에서 GPS(위치) 정보만 제거합니다. 실패하면 None(메타데이터 전체 제거)을 돌려줍니다."""
    if not exif_bytes:
        return None
    try:
        exif = Image.Exif()
        exif.load(exif_bytes)
        # 34853 = GPSInfo IFD 포인터
        try:
            exif.pop(34853, None)
        except Exception:
            pass
        try:
            exif.pop(0x8825, None)
        except Exception:
            pass
        return exif.tobytes()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1) Pre-flight 진단
# ---------------------------------------------------------------------------
def _box_blur(a: np.ndarray, radius: int = 1) -> np.ndarray:
    """float32 2D 배열의 분리형 박스 블러(엣지 패딩)."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    p = np.pad(a, ((radius, radius), (radius, radius)), mode="edge")
    cs = np.cumsum(p, axis=0, dtype=np.float32)
    cs = np.vstack([np.zeros((1, cs.shape[1]), np.float32), cs])
    out = (cs[k:] - cs[:-k]) / k
    cs = np.cumsum(out, axis=1, dtype=np.float32)
    cs = np.hstack([np.zeros((cs.shape[0], 1), np.float32), cs])
    return (cs[:, k:] - cs[:, :-k]) / k


def _laplacian_sigma(gray01: np.ndarray) -> float:
    lap = (4.0 * gray01[1:-1, 1:-1]
           - gray01[:-2, 1:-1] - gray01[2:, 1:-1]
           - gray01[1:-1, :-2] - gray01[1:-1, 2:])
    return float(lap.std())


def _noise_sigma(gray01: np.ndarray) -> float:
    """평탄 영역의 고주파 잔차(MAD 기반)로 노이즈를 추정합니다(0~1 스케일).

    표준편차는 디테일 많은 사진에서 과대평가되므로, 하위 15% 평탄 영역의
    중위절대편차(MAD)를 씨서 질감과 노이즈를 구분합니다.
    """
    blur = _box_blur(gray01, 1)
    resid = gray01 - blur
    gy, gx = np.gradient(blur)
    grad = np.hypot(gx, gy)
    flat = grad <= float(np.percentile(grad, 15))
    sample = resid[flat] if int(flat.sum()) >= 200 else resid.ravel()
    return float(np.median(np.abs(sample)) * 1.4826)


def _chroma_noise_sigma(arr: np.ndarray) -> float:
    """Cb/Cr 채널의 고주파 잔차(MAD)로 색 노이즈를 추정합니다(0~1 스케일)."""
    h, w = arr.shape[:2]
    step = max(1, int(math.sqrt((h * w) / 1_500_000)))
    sub = arr[::step, ::step].astype(np.float32) / 255.0
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    vals = []
    for ch in (cb, cr):
        blur = _box_blur(ch, 1)
        resid = ch - blur
        gy, gx = np.gradient(blur)
        grad = np.hypot(gx, gy)
        flat = grad <= float(np.percentile(grad, 15))
        sample = resid[flat] if int(flat.sum()) >= 200 else resid.ravel()
        vals.append(float(np.median(np.abs(sample)) * 1.4826))
    return float(np.mean(vals))


def measure_image(img: Image.Image, info: Optional[Dict[str, Any]] = None,
                  source_size_bytes: Optional[int] = None) -> Dict[str, Any]:
    """로컬 정량 지표를 모두 계산해 딕셔너리로 돌려줍니다."""
    info = info or {}
    img = img.convert("RGB") if img.mode != "RGB" else img
    w, h = img.size
    mp = (w * h) / 1_000_000.0

    arr = np.asarray(img, dtype=np.uint8)
    gray_full = np.asarray(img.convert("L"), dtype=np.uint8)   # 8bit 휘도(대용량에서도 가벼움)

    # 히스토그램 기반 지표(대형 이미지는 서브샘플)
    step = max(1, int(math.sqrt((w * h) / 2_000_000)))
    lum_sub = gray_full[::step, ::step]
    mean_luma = float(lum_sub.mean() / 255.0)
    p01, p50, p99 = (float(np.percentile(lum_sub, p) / 255.0) for p in (1, 50, 99))
    highlight_pct = float((lum_sub >= 250).mean() * 100.0)
    shadow_pct = float((lum_sub <= 4).mean() * 100.0)
    pure_white_pct = float((arr[::step, ::step].min(axis=2) >= 254).mean() * 100.0)

    # 선명도: 해상도 의존성을 줄이기 위해 장변 1600px 로 정규화한 뒤 측정
    scale = min(1.0, 1600.0 / max(w, h))
    if scale < 1.0:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        gray_small = np.asarray(img.convert("L").resize((sw, sh), Image.LANCZOS), dtype=np.float32) / 255.0
    else:
        gray_small = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    lap_sigma = _laplacian_sigma(gray_small)
    lap_var = lap_sigma ** 2

    # 노이즈: 네이티브 해상도 중앙 크롭(최대 1400px)
    cw, ch = min(w, 1400), min(h, 1400)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    crop_gray = gray_full[y0:y0 + ch, x0:x0 + cw].astype(np.float32) / 255.0
    noise_sigma = _noise_sigma(crop_gray)
    chroma_sigma = _chroma_noise_sigma(arr[y0:y0 + ch, x0:x0 + cw])

    # 색 성향
    sub = arr[::step, ::step].astype(np.float32)
    r_mean, g_mean, b_mean = (float(sub[:, :, i].mean()) for i in range(3))
    gray_ref = (0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean) or 1.0
    cast = {
        "r": (r_mean - gray_ref) / gray_ref,
        "g": (g_mean - gray_ref) / gray_ref,
        "b": (b_mean - gray_ref) / gray_ref,
    }
    max_cast = max(abs(v) for v in cast.values())

    mx = sub.max(axis=2)
    mn = sub.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    sat_mean = float(sat.mean())
    hot_pct = float((sat > 0.92).mean() * 100.0)

    file_mb = (source_size_bytes / (1024 * 1024)) if source_size_bytes else None

    metrics: Dict[str, Any] = {
        "width": w,
        "height": h,
        "megapixels": mp,
        "aspect_label": _gcd_ratio(w, h),
        "aspect_value": round(w / h, 4),
        "orientation": "가로" if w >= h else "세로",
        "min_side": min(w, h),
        "max_side": max(w, h),
        "format": info.get("format", "UNKNOWN"),
        "source_mode": info.get("source_mode", ""),
        "icc_name": info.get("icc_name"),
        "is_srgb": info.get("is_srgb", True),
        "has_exif": info.get("has_exif", False),
        "file_size_mb": file_mb,
        "mean_luma": round(mean_luma, 4),
        "median_luma": round(p50, 4),
        "p01_luma": round(p01, 4),
        "p99_luma": round(p99, 4),
        "dynamic_range": round(p99 - p01, 4),
        "highlight_clip_pct": round(highlight_pct, 3),
        "pure_white_pct": round(pure_white_pct, 3),
        "shadow_clip_pct": round(shadow_pct, 3),
        "laplacian_sigma": round(lap_sigma, 5),
        "laplacian_var": round(lap_var, 8),
        "noise_sigma": round(noise_sigma, 5),
        "chroma_noise_sigma": round(chroma_sigma, 5),
        "saturation_mean": round(sat_mean, 4),
        "oversaturated_pct": round(hot_pct, 3),
        "color_cast": {k: round(v, 4) for k, v in cast.items()},
        "color_cast_max": round(max_cast, 4),
    }
    metrics["sharpness_score"] = _sharpness_score(lap_sigma)
    metrics["noise_score"] = _noise_score(noise_sigma, chroma_sigma)
    return metrics


def _sharpness_score(lap_sigma: float) -> int:
    """라플라시안 표준편차(장변 1600px 정규화) → 0~100 선명도 점수."""
    # 실측 기준: 가우시안 흐림 3px ≈ 0.007, 약간 무름 ≈ 0.02, 보통 0.045, 선명 0.09+
    lo, hi = 0.008, 0.090
    v = math.log10(max(lap_sigma, 1e-6))
    lo_l, hi_l = math.log10(lo), math.log10(hi)
    score = (v - lo_l) / (hi_l - lo_l) * 100.0
    return int(round(clamp(score, 0, 100)))


def _noise_score(noise_sigma: float, chroma_sigma: float) -> int:
    """노이즈(0~1 MAD 잔차) → 0~100 '깨끗함' 점수(높을수록 깨끗)."""
    # 8bit 환산: 0.4 미만은 매우 깨끗, 5.0 이상이면 심한 노이즈
    combined = (noise_sigma * 255.0) * 0.80 + (chroma_sigma * 255.0) * 0.50
    score = 100.0 - (combined - 0.40) / (5.00 - 0.40) * 100.0
    return int(round(clamp(score, 0, 100)))


def _status(sev: str) -> str:
    return {"pass": "통과", "warn": "주의", "fail": "위험", "info": "참고"}[sev]


def evaluate_preflight(metrics: Dict[str, Any], filename: str = "") -> Dict[str, Any]:
    """측정 지표를 스톡 심사 기준 체크리스트로 평가합니다."""
    checks: List[Dict[str, Any]] = []

    def add(key: str, label: str, sev: str, value: str, advice: str = "") -> None:
        checks.append({"key": key, "label": label, "severity": sev,
                       "status": _status(sev), "value": value, "advice": advice})

    w, h = metrics["width"], metrics["height"]
    mp = metrics["megapixels"]
    min_side = metrics["min_side"]

    # 1. 해상도
    if mp < ADOBE_MIN_MEGAPIXELS:
        add("resolution", "해상도 (최소 4 MP)",
            "fail", f"{mp:.2f} MP ({w}×{h})",
            "Adobe Stock 사진 최소 규격 4MP 미만입니다. 원본 파일로 다시 제출하거나 "
            "더 큰 해상도로 촬영/현상하세요. 업스케일은 화질 저하로 반려될 수 있습니다.")
    elif mp < ADOBE_RECOMMENDED_MEGAPIXELS:
        add("resolution", "해상도 (최소 4 MP)",
            "warn", f"{mp:.2f} MP ({w}×{h})",
            "최소 규격은 통과하지만 여유가 적습니다. 크롭 여유를 위해 6MP 이상을 권장합니다.")
    else:
        add("resolution", "해상도 (최소 4 MP)", "pass", f"{mp:.2f} MP ({w}×{h})",
            "심사 규격을 충분히 상회합니다.")

    # 2. 변 길이
    if min_side < ADOBE_MIN_SIDE:
        add("min_side", "최소 변 길이 (권장 1,600px)", "warn",
            f"{min_side}px ({w}×{h})",
            "짧은 변이 1,600px 미만이면 확대 사용 시 화질 문제가 생깁니다. "
            "가능하면 더 큰 원본을 사용하세요.")
    else:
        add("min_side", "최소 변 길이 (권장 1,600px)", "pass", f"{min_side}px")

    # 3. 종횡비
    aspect = metrics["aspect_value"]
    if aspect > 4.0 or aspect < 0.25:
        add("aspect", "종횡비", "info",
            f"{metrics['aspect_label']} ({aspect:.2f}:1)",
            "매우 긴 파노라마 비율입니다. 스톡에서는 정상 통과하지만 썸네일에서 잘릴 수 있습니다.")
    else:
        add("aspect", "종횡비", "pass", f"{metrics['aspect_label']} ({metrics['orientation']} {w}×{h})")

    # 4. 하이라이트 클리핑
    hp = metrics["highlight_clip_pct"]
    if hp > 6.0:
        add("highlight", "하이라이트 날아감", "fail", f"{hp:.2f}%",
            "'노출 과다(Overexposed)' 반려 사유입니다. 하이라이트 보호를 올려 다시 보정하거나, "
            "RAW 원본에서 노출을 -0.5EV 내려 재현하세요.")
    elif hp > 2.0:
        add("highlight", "하이라이트 날아감", "warn", f"{hp:.2f}%",
            "흰 영역이 살짝 넘칩니다. 하이라이트 보호(0.5~0.7)로 디테일을 살리세요.")
    else:
        add("highlight", "하이라이트 날아감", "pass", f"{hp:.2f}%")

    # 5. 암부 클리핑
    sp = metrics["shadow_clip_pct"]
    if sp > 20.0:
        add("shadow", "암부 뭉개짐", "fail", f"{sp:.2f}%",
            "'노출 부족(Underexposed)' 판정 위험. 암부 리프트를 올려 디테일을 복원하세요.")
    elif sp > 8.0:
        add("shadow", "암부 뭉개짐", "warn", f"{sp:.2f}%",
            "어두운 영역이 넓게 뭉개졌습니다. 암부 리프트 0.3~0.5 를 적용해 보세요.")
    else:
        add("shadow", "암부 뭉개짐", "pass", f"{sp:.2f}%")

    # 6. 선명도
    ss = metrics["sharpness_score"]
    if ss < 25:
        add("sharpness", "선명도 (Laplacian)", "fail",
            f"{ss}/100 (σ={metrics['laplacian_sigma']})",
            "초점 흐림/손떨림 반려 위험. 보정으로 해결되지 않습니다 — 원본 재촬영이 정답입니다.")
    elif ss < 45:
        add("sharpness", "선명도 (Laplacian)", "warn",
            f"{ss}/100 (σ={metrics['laplacian_sigma']})",
            "약간 무릅니다. 안전 샤프닝을 0.35~0.5 로 적용해 보세요.")
    else:
        add("sharpness", "선명도 (Laplacian)", "pass", f"{ss}/100 (σ={metrics['laplacian_sigma']})")

    # 7. 노이즈
    ns = metrics["noise_score"]
    raw_noise = metrics["noise_sigma"] * 255
    if ns < 25:
        add("noise", "노이즈", "fail", f"{ns}/100 (휘도 σ={raw_noise:.2f})",
            "노이즈로 '화질 불량(Noise)' 반려 위험. 크로마 노이즈 억제를 올리고, "
            "저ISO 원본 사용을 권장합니다.")
    elif ns < 50:
        add("noise", "노이즈", "warn", f"{ns}/100 (휘도 σ={raw_noise:.2f})",
            "노이즈가 보입니다. 크로마 노이즈 억제 0.3~0.5 로 정돈하세요.")
    else:
        add("noise", "노이즈", "pass", f"{ns}/100 (휘도 σ={raw_noise:.2f})")

    # 8. 과채도
    over = metrics["oversaturated_pct"]
    if over > 3.0:
        add("saturation", "과채도", "warn", f"고채도 픽셀 {over:.2f}%",
            "'과도한 보정(Over-processed)' 반려 위험. 생동감을 낮추고 채도 상한을 조이세요.")
    else:
        add("saturation", "과채도", "pass", f"고채도 픽셀 {over:.2f}%")

    # 9. 색공간
    if not metrics["is_srgb"]:
        add("colorspace", "색공간", "warn", f"{metrics['icc_name'] or '비 sRGB'}",
            "Adobe Stock 은 sRGB 를 기준으로 합니다. 앱에서 'sRGB로 변환'을 켜고 내보내세요.")
    else:
        add("colorspace", "색공간", "pass", metrics["icc_name"] or "sRGB(프로필 없음)")

    # 10. 파일 형식
    fmt = metrics["format"]
    if fmt in ("JPEG", "JPG"):
        add("format", "파일 형식", "pass", fmt)
    else:
        add("format", "파일 형식", "warn", fmt,
            "Adobe Stock 사진은 JPEG 제출이 표준입니다. 이 앱이 JPEG 로 변환해 저장합니다.")

    # 11. 파일 용량
    fmb = metrics["file_size_mb"]
    if fmb is None:
        add("filesize", "파일 용량 (최대 100MB)", "info", "미확인")
    elif fmb > ADOBE_MAX_FILE_MB:
        add("filesize", "파일 용량 (최대 100MB)", "fail", f"{fmb:.1f} MB",
            "100MB 초과 파일은 업로드되지 않습니다. 품질 95 로 다시 저장하세요.")
    else:
        add("filesize", "파일 용량 (최대 100MB)", "pass", f"{fmb:.1f} MB")

    # 12. 파일명
    if filename:
        base = os.path.basename(filename)
        if any(ord(c) > 127 for c in base):
            add("filename", "파일명", "warn", base,
                "한글/특수문자 파일명은 CSV 매칭 오류를 만듭니다. 영문·숫자·_ 로 바꾸세요.")
        else:
            add("filename", "파일명", "pass", base)

    # 13. 로고/초상권 — 로컬 판별 불가 안내
    add("rights", "로고·상표·초상권", "info", "로컬 판별 불가",
        "AI(Gemini) 심사 리스크 점검이나 육안 확인이 필요합니다. "
        "브랜드 로고·차량 번호판·얼굴·사유 건물은 릴리즈 필요 여부를 확인하세요.")

    counts = {sev: sum(1 for c in checks if c["severity"] == sev) for sev in ("pass", "warn", "fail", "info")}

    # 점수: 위험 -18, 주의 -7 (단, 참고는 감점 없음)
    score = 100 - counts["fail"] * 18 - counts["warn"] * 7
    score = int(clamp(score, 0, 100))
    if counts["fail"]:
        verdict, verdict_ko = "fail", "지금 상태로는 반려 가능성이 큽니다"
    elif counts["warn"] >= 3:
        verdict, verdict_ko = "warn", "통과 가능하지만 손볼 곳이 몇 군데 있습니다"
    elif counts["warn"]:
        verdict, verdict_ko = "caution", "거의 통과 수준입니다 (주의 항목 1~2개)"
    else:
        verdict, verdict_ko = "pass", "스톡 심사 규격에 잘 맞습니다"

    return {
        "score": score,
        "verdict": verdict,
        "verdict_ko": verdict_ko,
        "counts": counts,
        "checks": checks,
    }


def analyze(source: Any, filename: str = "", source_size_bytes: Optional[int] = None,
            to_srgb: bool = False) -> Dict[str, Any]:
    """이미지 → (PIL 이미지, 메타정보, 지표, 체크리스트) 한 번에."""
    img, info = load_image(source)
    if to_srgb and info.get("icc_profile"):
        img = convert_to_srgb(img, info.get("icc_profile"))
    if source_size_bytes is None and isinstance(source, (str, os.PathLike)):
        try:
            source_size_bytes = os.path.getsize(source)
        except OSError:
            source_size_bytes = None
    metrics = measure_image(img, info, source_size_bytes)
    report = evaluate_preflight(metrics, filename)
    return {"image": img, "info": info, "metrics": metrics, "report": report}


# ---------------------------------------------------------------------------
# 2) 보정 엔진
# ---------------------------------------------------------------------------
def _white_balance_gains(arr: np.ndarray, strength: float) -> Tuple[float, float, float]:
    """회색 세계(gray-world) 가정으로 채널 게인을 계산합니다. 강도만큼만 적용."""
    h, w = arr.shape[:2]
    step = max(1, int(math.sqrt((h * w) / 400_000)))
    sub = arr[::step, ::step].astype(np.float32)
    flat = sub.reshape(-1, 3)
    lum = flat.mean(axis=1)
    # 중간 톤만 사용해 하늘/그림자 편향을 줄임
    mask = (lum > 30) & (lum < 235)
    sample = flat[mask] if int(mask.sum()) > 1000 else flat
    means = sample.mean(axis=0)
    means = np.maximum(means, 1.0)
    target = float(means.mean())
    gains = target / means
    gains = np.clip(gains, 0.82, 1.22)
    gains = 1.0 + (gains - 1.0) * float(strength)
    return float(gains[0]), float(gains[1]), float(gains[2])


def _tone_curve(x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """노출 → 하이라이트 보호 → 암부 리프트 → 블랙 포인트 → S자 대비 순서의 톤 커브."""
    y = x.copy()

    exposure = p["exposure"]
    if abs(exposure) > 1e-6:
        y = y * (2.0 ** exposure)

    hp = p["highlight_protect"]
    if hp > 1e-6:
        knee = 0.78
        k = 1.0 + 1.15 * hp
        z = np.clip((y - knee) / (1.0 - knee), 0.0, None)
        y = np.where(y > knee, knee + (1.0 - knee) * np.power(z, k), y)

    sl = p["shadow_lift"]
    if sl > 1e-6:
        g = 1.0 / (1.0 + 0.95 * sl)
        lifted = np.power(np.clip(y, 0.0, 1.0), g)
        mask = np.clip((0.5 - y) / 0.5, 0.0, 1.0) ** 1.4
        y = y + (lifted - y) * mask

    bp = p["black_point"]
    if bp > 1e-6:
        y = np.clip((y - bp) / (1.0 - bp), 0.0, 1.0)

    c = p["contrast"]
    s = (c - 1.0) * 2.2          # S자 대비: 0과 1을 고정하므로 클리핑을 새로 만들지 않음
    if abs(s) > 1e-6:
        y = y + s * (y * (1.0 - y) * (2.0 * y - 1.0))

    # 웜/쿨 미세 틸트는 채널별 LUT 에서 처리하므로 여기서는 톤만
    return np.clip(y, 0.0, 1.0)


def _apply_wb_tone(arr_u8: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    """화이트밸런스 + 톤 커브를 채널별 256단 LUT 로 적용합니다(8bit 고속 경로)."""
    gains = list(_white_balance_gains(arr_u8, p["wb_strength"]))
    warm = p["warmth"]
    gains[0] *= (1.0 + warm * 6.0)
    gains[2] *= (1.0 - warm * 6.0)

    levels = np.arange(256, dtype=np.float32) / 255.0
    out = np.empty_like(arr_u8)
    for c in range(3):
        shifted = np.clip(levels * gains[c], 0.0, 1.0)
        lut = (_tone_curve(shifted, p) * 255.0 + 0.5).astype(np.uint8)
        np.take(lut, arr_u8[:, :, c], out=out[:, :, c])
    return out


def _apply_vibrance(arr_f: np.ndarray, vibrance: float, sat_cap: float) -> np.ndarray:
    """저채도 픽셀 위주로 생동감을 올리고, 채도 상한으로 과채도를 막습니다.

    메모리·속도를 위해 arr_f 를 제자리에서(in-place) 수정하며,
    axis=2 축소(max/min)와 불리언 인덱싱 대신 채널별 요소 연산을 씁니다(대용량 사진에서 훨씬 빠름).
    """
    if vibrance > 1e-6:
        r_ch, g_ch, b_ch = arr_f[:, :, 0], arr_f[:, :, 1], arr_f[:, :, 2]
        mx = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        mn = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        sat = (mx - mn) / np.maximum(mx, 1e-6)
        np.clip(sat, 0.0, 1.0, out=sat)
        weight = 1.0 - sat
        del sat
        weight *= weight                                  # (1-sat)^2 : 저채도 위주 가중
        weight *= (vibrance * 1.15)
        weight += 1.0
        gray = arr_f.mean(axis=2, dtype=np.float32)       # 중립축(채널 평균)
        for c in range(3):                                # 채널별 제자리 연산(대형 임시배열 회피)
            ch = arr_f[:, :, c]
            ch -= gray
            ch *= weight
            ch += gray
        del mx, mn, gray, weight
        np.clip(arr_f, 0.0, 1.0, out=arr_f)

    # 채도 상한: 상한을 넘는 픽셀만 회색 쪽으로 압축 (클리핑 없이)
    if sat_cap < 0.999:
        mx2 = np.maximum(np.maximum(arr_f[:, :, 0], arr_f[:, :, 1]), arr_f[:, :, 2])
        mn2 = np.minimum(np.minimum(arr_f[:, :, 0], arr_f[:, :, 1]), arr_f[:, :, 2])
        sat2 = (mx2 - mn2) / np.maximum(mx2, 1e-6)
        if float(sat2.max()) > sat_cap:
            scale = sat_cap / np.maximum(sat2, 1e-6)
            np.minimum(scale, 1.0, out=scale)             # 상한 이하 픽셀은 1.0 그대로
            del sat2
            gray2 = arr_f.mean(axis=2, dtype=np.float32)
            for c in range(3):
                ch = arr_f[:, :, c]
                ch -= gray2
                ch *= scale
                ch += gray2
            np.clip(arr_f, 0.0, 1.0, out=arr_f)
            del gray2, scale
        del mx2, mn2
    return arr_f


def _apply_chroma_nr(img: Image.Image, strength: float) -> Image.Image:
    """휘도는 그대로 두고 색차(Cb/Cr)만 반해상도에서 부드럽게 만들어 색 노이즈를 줄입니다.

    32비트 float 전면 배열을 만들지 않고 8bit(PIL) 경로로 처리해 대용량 사진에서도 메모리가 작습니다.
    """
    if strength <= 1e-6:
        return img
    small_size = (max(1, img.width // 2), max(1, img.height // 2))
    ycbcr = img.resize(small_size, Image.BILINEAR).convert("YCbCr")
    _y, cb, cr = ycbcr.split()
    radius = 0.8 + 1.6 * strength
    cb = Image.blend(cb, cb.filter(ImageFilter.GaussianBlur(radius)), clamp(strength, 0.0, 1.0))
    cr = Image.blend(cr, cr.filter(ImageFilter.GaussianBlur(radius)), clamp(strength, 0.0, 1.0))
    cb_full = cb.resize(img.size, Image.BILINEAR)
    cr_full = cr.resize(img.size, Image.BILINEAR)
    y_full = img.convert("YCbCr").split()[0]      # 원본 휘도 = 디테일 보존
    return Image.merge("YCbCr", (y_full, cb_full, cr_full)).convert("RGB")


def _apply_safe_sharpen(arr_f: np.ndarray, amount: float, threshold: float) -> np.ndarray:
    """휘도만 샤프닝하고, 강한 엣지(할로 발생 지점)는 임계값으로 건너뜁니다.

    arr_f 를 제자리에서 수정하고 중간 배열을 재사용해 대용량 사진에서도 메모리가 작습니다.
    """
    if amount <= 1e-6:
        return arr_f
    # 휘도(0~1)
    y = arr_f[:, :, 0] * 0.299 + arr_f[:, :, 1] * 0.587 + arr_f[:, :, 2] * 0.114
    y8 = Image.fromarray((np.clip(y, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L")
    blur = np.asarray(y8.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    blur /= 255.0
    del y8

    detail = y - blur                       # 디테일(고주파)
    del blur
    keep = np.abs(detail)                   # 할로 가드용 가중치 → 같은 배열 재사용
    keep /= max(threshold * 4.0, 1e-4)
    np.clip(keep, 0.0, 1.0, out=keep)
    np.subtract(1.0, keep, out=keep)        # 강한 엣지일수록 0 에 가깝게
    detail *= (amount * 1.6)
    detail *= keep
    del keep
    np.clip(detail, -0.05, 0.05, out=detail)

    # 상대 부스트 → 채널 게인 (색비 유지, y=0 픽셀은 그대로)
    detail /= np.maximum(y, 1e-4)
    detail += 1.0
    detail[y < 1e-4] = 1.0
    for c in range(3):
        arr_f[:, :, c] *= detail
    del detail, y
    np.clip(arr_f, 0.0, 1.0, out=arr_f)
    return arr_f


def enhance(img: Image.Image, params: Optional[Dict[str, Any]] = None,
            guard: bool = True, max_workers: int = 1) -> Dict[str, Any]:
    """
    스톡 전용 보정을 적용합니다.

    반환: {"image": PIL, "params": 실제 적용 파라미터, "stats": 처리 요약}
    """
    p = sanitize_params(params, guard=guard)
    work = ImageOps.exif_transpose(img.convert("RGB")) if img.mode != "RGB" else img
    arr_u8 = np.asarray(work, dtype=np.uint8)

    # (1) 화이트밸런스 + 노출 + 하이라이트/암부 + 대비  → 채널별 LUT
    arr_u8 = _apply_wb_tone(arr_u8, p)

    # (2) 생동감/채도 상한 (float, 제자리 연산)
    need_vib = p["vibrance"] > 1e-6 or p["saturation_cap"] < 0.999
    if need_vib:
        arr_f = arr_u8.astype(np.float32)
        arr_f /= 255.0
        _apply_vibrance(arr_f, p["vibrance"], p["saturation_cap"])
        arr_f *= 255.0
        arr_f += 0.5
        np.clip(arr_f, 0.0, 255.0, out=arr_f)
        arr_u8 = arr_f.astype(np.uint8)
        del arr_f

    out_img = Image.fromarray(arr_u8)

    # (3) 크로마 노이즈 억제 (8bit/PIL 경로 — 휘도 디테일 보존)
    if p["chroma_nr"] > 1e-6:
        out_img = _apply_chroma_nr(out_img, p["chroma_nr"])
        arr_u8 = np.asarray(out_img, dtype=np.uint8)

    # (4) 안전 샤프닝(할로 방지)
    if p["sharpen"] > 1e-6:
        arr_f = arr_u8.astype(np.float32)
        arr_f /= 255.0
        _apply_safe_sharpen(arr_f, p["sharpen"], p["sharpen_threshold"])
        arr_f *= 255.0
        arr_f += 0.5
        np.clip(arr_f, 0.0, 255.0, out=arr_f)
        out_img = Image.fromarray(arr_f.astype(np.uint8))
        del arr_f

    # 보정 후 클리핑 통계 — 8bit 휘도(PIL L)로 가볍게 계산
    gray8 = np.asarray(out_img.convert("L"), dtype=np.uint8)
    clip_stats = (round(float((gray8 >= 250).mean() * 100.0), 3),
                  round(float((gray8 <= 4).mean() * 100.0), 3))
    del gray8

    return {
        "image": out_img,
        "params": p,
        "stats": {
            "size": out_img.size,
            "highlight_clip_pct": clip_stats[0],
            "shadow_clip_pct": clip_stats[1],
        },
    }


def auto_tune(metrics: Dict[str, Any], preset_key: str, guard: bool = True) -> Dict[str, float]:
    """
    프리셋을 출발점으로, 이미지 지표에 맞춰 파라미터를 자동으로 다듬습니다.
    (Gemini 없이도 '사진 보정 전문가' 수준의 판단을 흉내내는 규칙 기반 튜닝)
    """
    p = preset_params(preset_key, guard=guard)

    hp = metrics["highlight_clip_pct"]
    sp = metrics["shadow_clip_pct"]
    mean_l = metrics["mean_luma"]
    ss = metrics["sharpness_score"]
    ns = metrics["noise_score"]
    sat = metrics["saturation_mean"]
    cast = metrics["color_cast_max"]

    # 클리핑이 있으면 하이라이트 보호 강화 + 노출 하향
    if hp > 4.0:
        p["highlight_protect"] = min(1.0, p["highlight_protect"] + 0.40)
        p["exposure"] -= 0.18
    elif hp > 1.5:
        p["highlight_protect"] = min(1.0, p["highlight_protect"] + 0.20)
        p["exposure"] -= 0.07

    # 암부가 뭉개졌으면 리프트 강화
    if sp > 12.0:
        p["shadow_lift"] = min(1.0, p["shadow_lift"] + 0.28)
    elif sp > 5.0:
        p["shadow_lift"] = min(1.0, p["shadow_lift"] + 0.12)

    # 전반적 노출 보정
    if mean_l < 0.30:
        p["exposure"] += 0.22
    elif mean_l < 0.40:
        p["exposure"] += 0.10
    elif mean_l > 0.72:
        p["exposure"] -= 0.12
    elif mean_l > 0.64:
        p["exposure"] -= 0.06

    # 흐리면 샤프닝을 조금 올리되 상한 유지
    if ss < 35:
        p["sharpen"] = min(p["sharpen"] + 0.18, 0.55)
    elif ss > 70:
        p["sharpen"] = max(p["sharpen"] - 0.08, 0.12)

    # 노이즈가 많으면 샤프닝을 줄이고 색노이즈 억제를 올린다
    if ns < 40:
        p["sharpen"] = max(p["sharpen"] - 0.12, 0.10)
        p["chroma_nr"] = min(1.0, p["chroma_nr"] + 0.22)
        p["sharpen_threshold"] = min(0.10, p["sharpen_threshold"] + 0.015)

    # 이미 채도가 높으면 생동감을 낮춘다
    if sat > 0.55:
        p["vibrance"] = max(0.0, p["vibrance"] - 0.08)
        p["saturation_cap"] = max(0.72, p["saturation_cap"] - 0.05)
    elif sat < 0.18:
        p["vibrance"] = min(0.40, p["vibrance"] + 0.08)

    # 색 편향이 크면 화이트밸런스 강도를 올린다
    if cast > 0.035:
        p["wb_strength"] = min(1.0, p["wb_strength"] + 0.20)
        p["warmth"] = 0.0  # 자동 판단 시 임의 틸트는 넣지 않는다

    return sanitize_params(p, guard=guard)


# ---------------------------------------------------------------------------
# 3) 저장 / 패키징 유틸
# ---------------------------------------------------------------------------
def save_jpeg(img: Image.Image, target: Any, quality: int = 96, subsampling: int = 0,
              exif_bytes: Optional[bytes] = None, icc_bytes: Optional[bytes] = None,
              keep_metadata: bool = True) -> None:
    """스톡 제출용 JPEG 로 저장합니다(기본 quality 96, subsampling 0 = 4:4:4)."""
    kwargs: Dict[str, Any] = {
        "format": "JPEG",
        "quality": int(clamp(quality, JPEG_QUALITY_MIN, JPEG_QUALITY_MAX)),
        "subsampling": int(subsampling),
        "optimize": True,
        "progressive": False,
    }
    if keep_metadata:
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        if icc_bytes:
            kwargs["icc_profile"] = icc_bytes
    img.save(target, **kwargs)


def jpeg_bytes(img: Image.Image, quality: int = 96, subsampling: int = 0,
               exif_bytes: Optional[bytes] = None, icc_bytes: Optional[bytes] = None,
               keep_metadata: bool = True) -> bytes:
    buf = io.BytesIO()
    save_jpeg(img, buf, quality=quality, subsampling=subsampling,
              exif_bytes=exif_bytes, icc_bytes=icc_bytes, keep_metadata=keep_metadata)
    return buf.getvalue()


def stock_filename(name: str, suffix: str = "_stock") -> str:
    """원본 파일명을 영문/숫자/_ 기반 스톡 제출용 파일명으로 정리합니다."""
    base, _ = os.path.splitext(os.path.basename(str(name)))
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in base)
    safe = "_".join(part for part in safe.split("_") if part) or "image"
    return f"{safe}{suffix}.jpg"


def iter_image_files(folder: str, recursive: bool = False) -> List[str]:
    """폴더에서 처리 가능한 이미지 경로 목록."""
    found: List[str] = []
    if not folder or not os.path.isdir(folder):
        return found
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    found.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append(path)
    return found


def process_path(path: str, preset_key: str = "natural_commercial", use_auto_tune: bool = True,
                 params: Optional[Dict[str, Any]] = None, guard: bool = True,
                 quality: int = 96, subsampling: int = 0, keep_metadata: bool = True,
                 strip_location: bool = True, to_srgb: bool = True) -> Dict[str, Any]:
    """파일 1장: 진단 → 보정 → JPEG 바이트 생성. 배치에서 재사용."""
    with open(path, "rb") as fh:
        raw = fh.read()
    result = analyze(io.BytesIO(raw), filename=os.path.basename(path),
                     source_size_bytes=len(raw), to_srgb=to_srgb)
    img, info, metrics, report = result["image"], result["info"], result["metrics"], result["report"]

    effective = dict(params) if params else (
        auto_tune(metrics, preset_key, guard=guard) if use_auto_tune else preset_params(preset_key, guard=guard)
    )
    enhanced = enhance(img, effective, guard=guard)

    exif = info.get("exif_bytes")
    if keep_metadata:
        if strip_location:
            exif = strip_gps(exif)
    else:
        exif = None
    icc = info.get("icc_profile") if keep_metadata else None
    if to_srgb:
        icc = None  # sRGB 변환했으므로 원본 프로필은 버린다(뷰어가 sRGB 로 해석)

    data = jpeg_bytes(enhanced["image"], quality=quality, subsampling=subsampling,
                      exif_bytes=exif, icc_bytes=icc, keep_metadata=keep_metadata)

    return {
        "source_path": path,
        "filename": os.path.basename(path),
        "out_name": stock_filename(path),
        "metrics": metrics,
        "report": report,
        "params": enhanced["params"],
        "enhanced": enhanced["image"],
        "data": data,
        "bytes": len(data),
        "post_stats": enhanced["stats"],
        "info": {k: v for k, v in info.items() if k not in ("icc_profile", "exif_bytes")},
    }


# ---------------------------------------------------------------------------
# 4) Adobe Stock CSV
# ---------------------------------------------------------------------------
def adobe_category_number(category: Any) -> str:
    """카테고리 이름/번호를 Adobe Stock CSV 용 번호 문자열로 변환합니다."""
    if category is None or category == "":
        return ""
    if isinstance(category, (int, float)):
        return str(int(category))
    text = str(category).strip()
    if text.isdigit():
        return text
    return str(ADOBE_CATEGORY_MAP.get(text.lower(), ""))


def build_adobe_csv(rows: Sequence[Dict[str, Any]]) -> str:
    """
    Adobe Stock 일괄 업로드 CSV.
    헤더: Filename,Title,Keywords,Category,Releases
    Keywords 는 쉼표로 구분하고 csv 모듈이 따옴표로 감쌉니다.
    """
    import csv

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Filename", "Title", "Keywords", "Category", "Releases"])
    for row in rows:
        keywords = row.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.replace(";", ",").split(",") if k.strip()]
        writer.writerow([
            row.get("filename", ""),
            (row.get("title") or "").strip(),
            ", ".join(str(k).strip() for k in keywords),
            adobe_category_number(row.get("category")),
            row.get("releases", "") or "",
        ])
    return buf.getvalue()


def build_shutterstock_csv(rows: Sequence[Dict[str, Any]]) -> str:
    """Shutterstock 벌크 업로드 CSV(참고용). Description/Keywords/Categories/Editorial."""
    import csv

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Filename", "Description", "Keywords", "Categories", "Editorial", "Mature content", "Illustration"])
    for row in rows:
        keywords = row.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.replace(";", ",").split(",") if k.strip()]
        writer.writerow([
            row.get("filename", ""),
            (row.get("title") or "").strip(),
            ", ".join(str(k).strip() for k in keywords[:50]),
            adobe_category_number(row.get("category")),
            "yes" if row.get("editorial") else "no",
            "no",
            "no",
        ])
    return buf.getvalue()
