"""
gemini_stock.py — 스톡 심사 리스크 점검 + 메타데이터 자동 생성 (선택 기능)

API 키가 없어도 앱의 모든 진단·보정 기능은 100% 동작합니다.
키를 넣었을 때만 이 모듈이 호출됩니다.

구글 Generative Language REST API 를 표준 라이브러리(urllib)로 직접 호출하므로
google-generativeai 패키지가 없어도 됩니다(설치 부담·버전 충돌 없음).
"""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from PIL import Image

from stock_engine import ADOBE_CATEGORIES

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"
# 모델명이 바뀌었거나 해당 키에서 쓸 수 없을 때 순서대로 시도
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

CATEGORY_NAMES = [name for _num, name in ADOBE_CATEGORIES]


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior Adobe Stock reviewer AND a top-selling stock contributor
who has reviewed over 100,000 submissions. You judge images strictly by stock agency
acceptance rules (technical quality, commercial usability, intellectual property risk)
and you write metadata the way high-earning contributors do: concrete, searchable,
no keyword spam, no hype, no brand names.

Analyse the attached photograph and return ONE JSON object only.
"""

TASK_PROMPT = """Return JSON with exactly these keys:

{
  "title_en": "Adobe Stock title, max 200 characters, plain descriptive English, no camera/technical jargon, no keywords stuffing, no quotes",
  "title_ko": "같은 제목의 자연스러운 한국어 번역",
  "description_en": "1-2 sentence factual description in English for Shutterstock-style description",
  "keywords_en": [ "25 to 30 single words or 2-word phrases in English, ordered most relevant first, lowercase, no duplicates, no '#'" ],
  "category": "exactly one of: <<CATEGORIES>>",
  "editorial_use": false,
  "risk_level": "low | medium | high",
  "risk_items": [ {"type": "logo | trademark | license_plate | face | identifiable_person | private_property | artwork | text | other", "severity": "low | medium | high", "where": "where in the frame", "advice": "concrete action the contributor must take (remove, blur, add release, submit as editorial)"} ],
  "release_needed": false,
  "summary_ko": "한국어 3~4문장: 이 사진이 스톡에서 팔릴 만한 이유, 심사에서 걸릴 만한 점, 보정/제출 전략",
  "quality_notes_ko": "한국어 1~2문장: 노출·색·선명도·구도 관점에서 보완점"
}

Rules:
- keywords_en must have between 25 and 30 items.
- If a commercial logo, brand name, or vehicle license plate is legible, add a risk item of that type.
- If a person's face is recognizable, add "identifiable_person" and set release_needed to true.
- If the image is newsworthy/editorial-only (event, protest, branded product close-up), set editorial_use true.
- Do not invent landmarks or locations you cannot see.
<<EXTRA>>
"""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class GeminiError(RuntimeError):
    """사용자에게 그대로 보여줄 수 있는 한국어 오류 메시지를 담습니다."""


def _post(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            pass
        detail = ""
        try:
            detail = json.loads(body).get("error", {}).get("message", "")
        except Exception:
            detail = body[:300]
        if e.code in (400, 401, 403):
            raise GeminiError(f"API 키가 올바르지 않거나 권한이 없습니다 ({e.code}). 키를 다시 확인해 주세요. {detail}") from e
        if e.code == 429:
            raise GeminiError("무료 사용량 한도(분당/일일)를 초과했습니다. 잠시 후 다시 시도하거나 다른 모델을 선택하세요.") from e
        if e.code == 404:
            raise GeminiError(f"모델을 찾을 수 없습니다 ({e.code}). 모델명을 확인하세요. {detail}") from e
        raise GeminiError(f"Gemini API 오류 {e.code}: {detail or '알 수 없는 오류'}") from e
    except (urllib.error.URLError, socket.timeout) as e:
        raise GeminiError(f"네트워크 연결에 실패했습니다: {e}. 인터넷 연결을 확인하세요.") from e


def _extract_text(response: Dict[str, Any]) -> str:
    try:
        candidates = response.get("candidates") or []
        parts = candidates[0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError):
        blocked = (response.get("promptFeedback") or {}).get("blockReason")
        if blocked:
            raise GeminiError(f"이미지가 안전 정책으로 차단되었습니다({blocked}). 다른 사진으로 시도해 주세요.")
        raise GeminiError("Gemini 응답에서 텍스트를 찾지 못했습니다.")


def _extract_json(text: str) -> Dict[str, Any]:
    """코드블록/설명 문장이 섞여 있어도 첫 JSON 객체를 찾아냅니다."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = cleaned[start:i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
    raise GeminiError("Gemini 응답을 JSON 으로 해석하지 못했습니다. 다시 시도해 주세요.")


def _prepare_image(image: Image.Image, max_side: int = 1280, quality: int = 88) -> tuple[bytes, str]:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "image/jpeg"


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def analyze_image(image: Image.Image, api_key: str, model: str = DEFAULT_MODEL,
                  local_metrics: Optional[Dict[str, Any]] = None,
                  extra_hint: str = "", timeout: int = 120) -> Dict[str, Any]:
    """
    스톡 심사 리스크 + 메타데이터를 생성합니다.

    반환: {"ok": True, "title_en":..., "keywords_en":[...], "category":..., ...}
          또는 {"ok": False, "error": "..."}
    """
    if not api_key or not str(api_key).strip():
        return {"ok": False, "error": "API 키가 없습니다. 로컬 진단·보정은 키 없이도 모두 동작합니다."}

    # 로컬 지표를 알려 주면 제목/키워드가 실제 사진 내용과 더 잘 맞습니다.
    metric_lines = ""
    if local_metrics:
        m = local_metrics
        metric_lines = (
            f"\nKnown technical facts measured locally (use them, do not contradict them):\n"
            f"- size {m.get('width')}x{m.get('height')} ({m.get('megapixels')} MP), aspect {m.get('aspect_label')}\n"
            f"- highlight clipping {m.get('highlight_clip_pct')}%, shadow clipping {m.get('shadow_clip_pct')}%, "
            f"mean luminance {m.get('mean_luma')}\n"
        )
    prompt = SYSTEM_PROMPT + "\n" + TASK_PROMPT.replace(
        "<<CATEGORIES>>", ", ".join(CATEGORY_NAMES)
    ).replace(
        "<<EXTRA>>", (metric_lines + (f"\nContributor note: {extra_hint}" if extra_hint else ""))
    )

    img_bytes, mime = _prepare_image(image)
    b64 = base64.b64encode(img_bytes).decode("ascii")

    models_to_try: List[str] = []
    for m in [model, *FALLBACK_MODELS]:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error: Optional[Exception] = None
    for candidate in models_to_try:
        url = f"{API_ROOT}/{candidate}:generateContent?key={api_key.strip()}"
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.35,
                "topP": 0.9,
                "maxOutputTokens": 3072,
                "responseMimeType": "application/json",
            },
        }
        try:
            response = _post(url, payload, timeout=timeout)
            text = _extract_text(response)
            data = _extract_json(text)
            normalized = _normalize(data)
            normalized["ok"] = True
            normalized["model_used"] = candidate
            return normalized
        except GeminiError as e:
            last_error = e
            msg = str(e)
            if "모델을 찾을 수 없습니다" in msg or " 404" in msg:
                continue  # 다음 모델로 재시도
            if "API 키" in msg or "한도" in msg or "네트워크" in msg:
                break     # 재시도 무의미
            continue
        except Exception as e:  # 예기치 못한 형식 오류 → 다음 모델 시도
            last_error = e
            continue

    return {"ok": False, "error": str(last_error) if last_error else "알 수 없는 오류"}


def test_connection(api_key: str, model: str = DEFAULT_MODEL, timeout: int = 30) -> Dict[str, Any]:
    """키/모델이 살아 있는지 텍스트 한 줄로 확인합니다."""
    if not api_key:
        return {"ok": False, "error": "API 키를 입력하세요."}
    for candidate in dict.fromkeys([model, *FALLBACK_MODELS]):
        url = f"{API_ROOT}/{candidate}:generateContent?key={api_key.strip()}"
        payload = {"contents": [{"role": "user", "parts": [{"text": "Reply with exactly: OK"}]}],
                   "generationConfig": {"temperature": 0, "maxOutputTokens": 16}}
        try:
            resp = _post(url, payload, timeout=timeout)
            return {"ok": True, "model_used": candidate,
                    "text": _extract_text(resp).strip()[:100]}
        except GeminiError as e:
            if "모델을 찾을 수 없습니다" in str(e):
                continue
            return {"ok": False, "error": str(e)}
        except Exception as e:
            continue
    return {"ok": False, "error": "사용 가능한 Gemini 모델을 찾지 못했습니다."}


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """응답을 앱이 쓰기 좋은 형태로 표준화합니다."""
    keywords = data.get("keywords_en") or data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in re.split(r"[,;]", keywords) if k.strip()]
    cleaned: List[str] = []
    seen = set()
    for k in keywords:
        k2 = str(k).strip().strip("#").strip().lower()
        k2 = re.sub(r"\s+", " ", k2)
        if not k2 or k2 in seen:
            continue
        seen.add(k2)
        cleaned.append(k2)
    cleaned = cleaned[:30]

    category = str(data.get("category") or "").strip()
    matched = ""
    for name in CATEGORY_NAMES:
        if category.lower() == name.lower():
            matched = name
            break
    if not matched:
        for name in CATEGORY_NAMES:
            if name.lower() in category.lower() or category.lower() in name.lower():
                matched = name
                break
    if not matched:
        matched = ""  # UI 에서 직접 고르도록 비워 둔다

    risk_items = data.get("risk_items") or []
    if not isinstance(risk_items, list):
        risk_items = []
    norm_items = []
    for item in risk_items[:8]:
        if not isinstance(item, dict):
            continue
        norm_items.append({
            "type": str(item.get("type", "other")).strip().lower(),
            "severity": str(item.get("severity", "low")).strip().lower(),
            "where": str(item.get("where", "")).strip(),
            "advice": str(item.get("advice", "")).strip(),
        })

    return {
        "title_en": str(data.get("title_en", "")).strip()[:200],
        "title_ko": str(data.get("title_ko", "")).strip(),
        "description_en": str(data.get("description_en", "")).strip(),
        "keywords_en": cleaned,
        "category": matched,
        "editorial_use": bool(data.get("editorial_use", False)),
        "risk_level": str(data.get("risk_level", "low")).strip().lower(),
        "risk_items": norm_items,
        "release_needed": bool(data.get("release_needed", False)),
        "summary_ko": str(data.get("summary_ko", "")).strip(),
        "quality_notes_ko": str(data.get("quality_notes_ko", "")).strip(),
        "raw": data,
    }
