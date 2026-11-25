"""
app.py — 스톡사진(Adobe Stock 등) 심사 기준 자동보정 스튜디오

실행:  streamlit run app.py
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import os
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

import stock_engine as se
import gemini_stock as gs

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="스톡사진 심사 통과 자동보정 스튜디오",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
VERDICT_COLOR = {"pass": "#1a7f37", "caution": "#9a6700", "warn": "#bf6d00", "fail": "#cf222e"}


# ---------------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------------
def img_block(image: Image.Image, caption: Optional[str] = None) -> None:
    """Streamlit 버전 차이를 흡수해서 폭 맞춤 이미지를 그립니다."""
    try:
        st.image(image, caption=caption, width="stretch")
    except Exception:  # 구버전 streamlit (width 문자열 미지원)
        st.image(image, caption=caption, use_container_width=True)


def df_block(df, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:  # 구버전 streamlit
        st.dataframe(df, use_container_width=True, **kwargs)


def digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analyze(data: bytes, name: str, to_srgb: bool) -> Dict[str, Any]:
    """진단 결과만 캐싱합니다(PIL 이미지는 매번 새로 디코드 — 대용량 사진 메모리 보호)."""
    result = se.analyze(io.BytesIO(data), filename=name, source_size_bytes=len(data), to_srgb=to_srgb)
    return {
        "info": {k: v for k, v in result["info"].items() if k not in ("icc_profile", "exif_bytes")},
        "metrics": result["metrics"],
        "report": result["report"],
    }


def open_display(data: bytes, to_srgb: bool) -> Image.Image:
    """표시용 이미지(원본). EXIF 회전·sRGB 변환을 적용합니다."""
    img, info = se.load_image(io.BytesIO(data))
    if to_srgb and info.get("icc_profile"):
        img = se.convert_to_srgb(img, info.get("icc_profile"))
    return img


@st.cache_data(show_spinner=False, max_entries=12)
def cached_enhance(data: bytes, params_key: Tuple[Tuple[str, float], ...],
                   quality: int, subsampling: int, to_srgb: bool,
                   keep_metadata: bool, strip_location: bool) -> Dict[str, Any]:
    """보정 결과를 JPEG 바이트로만 캐싱합니다."""
    img, info = se.load_image(io.BytesIO(data))
    if to_srgb and info.get("icc_profile"):
        img = se.convert_to_srgb(img, info.get("icc_profile"))
    params = dict(params_key)
    out = se.enhance(img, params, guard=True)

    exif = info.get("exif_bytes") if keep_metadata else None
    if keep_metadata and strip_location:
        exif = se.strip_gps(exif)
    icc = None if (to_srgb or not keep_metadata) else info.get("icc_profile")
    jpeg = se.jpeg_bytes(out["image"], quality=quality, subsampling=subsampling,
                         exif_bytes=exif, icc_bytes=icc, keep_metadata=keep_metadata)
    arr = np.asarray(out["image"], dtype=np.float32)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1.0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    sat_mean = round(float(sat.mean()), 4)
    del arr, mx, mn, sat
    return {
        "jpeg": jpeg,
        "params": out["params"],
        "stats": {**out["stats"], "saturation_mean": sat_mean},
    }


def check_table(report: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for c in report["checks"]:
        rows.append({
            "상태": f"{STATUS_ICON[c['severity']]} {c['status']}",
            "점검 항목": c["label"],
            "측정값": c["value"],
            "조치 / 안내": c["advice"] or "-",
        })
    return rows


def report_markdown(name: str, metrics: Dict[str, Any], report: Dict[str, Any],
                    params: Optional[Dict[str, float]] = None,
                    post: Optional[Dict[str, Any]] = None) -> str:
    lines = [f"# 스톡 심사 사전 진단 리포트 — {name}", "",
             f"- 생성: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"- 종합 점수: **{report['score']}/100** ({report['verdict_ko']})",
             f"- 해상도: {metrics['width']}×{metrics['height']} ({metrics['megapixels']:.2f} MP, {metrics['aspect_label']}, {metrics['orientation']})",
             f"- 하이라이트 날아감: {metrics['highlight_clip_pct']}% / 암부 뭉개짐: {metrics['shadow_clip_pct']}%",
             f"- 선명도: {metrics['sharpness_score']}/100 (Laplacian σ {metrics['laplacian_sigma']})", 
             f"- 노이즈: {metrics['noise_score']}/100 (휘도 σ {round(metrics['noise_sigma'] * 255, 2)})",
             ""]
    if post:
        lines += [f"- 보정 후 하이라이트 {post['highlight_clip_pct']}% / 암부 {post['shadow_clip_pct']}%", ""]
    lines += ["## 점검 결과", "", "| 상태 | 항목 | 측정값 | 조치 |", "|---|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['status']} | {c['label']} | {c['value']} | {c['advice'] or '-'} |")
    if params:
        lines += ["", "## 적용한 보정값", "", "| 파라미터 | 값 |", "|---|---|"]
        for k, v in params.items():
            lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
def _apply_preset_values(preset_key: str) -> None:
    """프리셋 값을 슬라이더 상태에 채워 넣습니다(콜백에서만 호출 → 위젯 재생성 안전)."""
    for k, v in se.preset_params(preset_key, guard=True).items():
        st.session_state[f"ft_{k}"] = float(v)


def _on_preset_change() -> None:
    picked = st.session_state.get("preset_widget", se.PRESET_ORDER[0])
    st.session_state["preset_key"] = picked
    _apply_preset_values(picked)


def _reset_to_preset() -> None:
    _apply_preset_values(st.session_state.get("preset_key", se.PRESET_ORDER[0]))


if not any(k.startswith("ft_") for k in st.session_state):
    _apply_preset_values(st.session_state.get("preset_key", se.PRESET_ORDER[0]))

with st.sidebar:
    st.markdown("### 📸 스톡 심사 통과 스튜디오")
    st.caption("Adobe Stock · Shutterstock 심사 기준에 맞춰 진단하고, "
               "과보정 없이 상업용으로 다듬습니다.")

    with st.expander("🔑 Gemini AI (선택)", expanded=False):
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIza... (없어도 모든 기능 동작)",
            help="https://aistudio.google.com/app/apikey 에서 무료 발급. "
                 "리스크 점검·제목·키워드·카테고리 자동 생성에만 사용됩니다.",
        )
        model = st.text_input("모델", value=gs.DEFAULT_MODEL,
                              help="모델이 없으면 자동으로 다른 모델을 시도합니다.")
        if st.button("연결 테스트", key="test_gemini"):
            with st.spinner("확인 중..."):
                res = gs.test_connection(api_key, model)
            if res.get("ok"):
                st.success(f"정상 ({res['model_used']})")
            else:
                st.error(res.get("error", "실패"))
        st.caption("키가 없어도 진단·보정·ZIP·CSV(수동 입력) 모두 정상 작동합니다.")

    st.markdown("---")
    st.markdown("#### 🎨 스톡 전용 프리셋")
    st.selectbox(
        "프리셋 선택",
        options=se.PRESET_ORDER,
        format_func=lambda k: f"{se.PRESETS[k]['label']} — {se.PRESETS[k]['label_ko']}",
        key="preset_widget",
        on_change=_on_preset_change,
    )
    preset_key = st.session_state.get("preset_widget", se.PRESET_ORDER[0])
    st.session_state["preset_key"] = preset_key
    st.caption(se.PRESETS[preset_key]["desc"])
    auto_tune = st.checkbox("사진 상태에 맞춰 자동 튜닝", value=True,
                            help="측정한 노출·클리핑·선명도·노이즈 지표에 맞춰 프리셋 값을 자동으로 조정합니다. "
                                 "슬라이더를 직접 움직이면 그 값이 우선합니다.")

    with st.expander("🛠️ 세부 미세조정 (스톡 안전 범위)", expanded=False):
        specs = [
            ("wb_strength", "자동 화이트밸런스 강도", 0.05, "회색 세계 가정으로 중립화합니다. 1.0 = 강하게 중립."),
            ("exposure", "노출 (EV)", 0.01, "±0.4EV 이내. 스톡은 과노출 반려가 가장 흔합니다."),
            ("highlight_protect", "하이라이트 보호", 0.05, "밝은 영역을 부드럽게 눌러 날아감을 막습니다."),
            ("shadow_lift", "암부 리프트", 0.05, "어두운 디테일 복원. 올리면 노이즈도 같이 살아납니다."),
            ("contrast", "대비 (S자 커브)", 0.01, "0과 1을 고정하는 S커브라 새로운 클리핑을 만들지 않습니다."),
            ("vibrance", "생동감 (Vibrance)", 0.02, "저채도 픽셀 위주로만 올려 과채도를 피합니다."),
            ("saturation_cap", "채도 상한", 0.01, "이 채도를 넘는 픽셀은 자동으로 눌러줍니다."),
            ("chroma_nr", "크로마 노이즈 억제", 0.05, "색 노이즈만 부드럽게. 휘도 디테일은 보존합니다."),
            ("sharpen", "안전 샤프닝", 0.02, "휘도만, 강한 엣지는 건너뛰는 할로 방지 샤프닝."),
            ("sharpen_threshold", "샤프닝 엣지 임계값", 0.005, "높을수록 엣지에서 샤프닝을 덜 합니다."),
            ("warmth", "색온도 미세 조정", 0.002, "+ 따뜻하게 / − 차갑게 (아주 미세하게만)."),
            ("black_point", "블랙 포인트", 0.001, "살짝 올려 검정을 정돈합니다."),
        ]
        for key, label, step, help_text in specs:
            lo, hi = se.GUARD_RANGES[key]
            st.slider(label, float(lo), float(hi), step=float(step),
                      key=f"ft_{key}", help=help_text)
        st.button("프리셋 값으로 되돌리기", on_click=_reset_to_preset)

    with st.expander("💾 출력 설정", expanded=True):
        quality = st.slider("JPEG 품질", 95, 98, 96,
                            help="스톡 제출 권장 95~98. subsampling 0(4:4:4)과 함께 씁니다.")
        subsampling_label = st.selectbox(
            "색상 서브샘플링", ["4:4:4 (최고 품질)", "4:2:2", "4:2:0 (용량 작음)"], index=0)
        subsampling = {"4:4:4 (최고 품질)": 0, "4:2:2": 1, "4:2:0 (용량 작음)": 2}[subsampling_label]
        meta_label = st.selectbox("메타데이터", ["촬영정보 유지 (GPS 제거)", "모두 유지", "모두 제거"], index=0,
                                  help="Adobe Stock 은 업로드 시 메타데이터를 자체 처리하지만, "
                                       "GPS 에 집 주소가 남는 것을 막으려면 첫 번째를 권장합니다.")
        keep_metadata = meta_label != "모두 제거"
        strip_location = meta_label == "촬영정보 유지 (GPS 제거)"
        to_srgb = st.checkbox("색공간을 sRGB 로 변환", value=True,
                              help="CMYK·AdobeRGB 등 비 sRGB 프로필을 sRGB 로 변환합니다(스톡 표준).")


# 현재 슬라이더 값 → 파라미터
current_params: Dict[str, float] = {}
for _k in se.PARAM_RANGES:
    current_params[_k] = float(st.session_state.get(f"ft_{_k}", se.DEFAULTS[_k]))
current_params = se.sanitize_params(current_params, guard=True)
params_key: Tuple[Tuple[str, float], ...] = tuple(sorted((k, round(v, 5)) for k, v in current_params.items()))


def resolve_params(metrics: Dict[str, Any]) -> Dict[str, float]:
    """자동 튜닝과 사용자 슬라이더 중 무엇을 쓸지 결정합니다.

    슬라이더가 프리셋 기본값 그대로면 → 사진 지표에 맞춘 자동 튜닝
    슬라이더를 직접 만졌으면      → 사용자가 정한 값(존중)
    """
    preset_defaults = tuple(sorted((k, round(v, 5)) for k, v in se.preset_params(preset_key).items()))
    if auto_tune and params_key == preset_defaults:
        return se.auto_tune(metrics, preset_key, guard=True)
    return current_params


# ---------------------------------------------------------------------------
# 헤더
# ---------------------------------------------------------------------------
st.title("📸 스톡사진 심사 통과 자동보정 스튜디오")
st.caption("Adobe Stock · Shutterstock **심사 기준 사전 진단** + **과보정 방지 보정 엔진** + "
           "**Adobe Stock CSV 내보내기**.  Gemini API 키 없이도 진단·보정·일괄처리·CSV 가 100% 동작합니다.")

tab_single, tab_batch, tab_help = st.tabs(["🔍 단일 사진 정밀 진단 · 보정", "🗂️ 다중 파일 / 폴더 일괄 보정", "📚 스톡 기준 · 사용법"])

# ===========================================================================
# TAB 1 — 단일 사진
# ===========================================================================
with tab_single:
    up = st.file_uploader("사진을 선택하세요 (JPG · PNG · TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"],
                          key="single_upload")

    if not up:
        st.info("사진을 올리면 **① 심사 사전 진단 → ② 프리셋 보정 → ③ Before/After 비교 → ④ CSV 내보내기** 순서로 진행됩니다.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**① Pre-flight 진단**")
            st.write("4MP 미만 경고, 하이라이트/암부 클리핑 %, Laplacian 선명도, 노이즈 추정, "
                     "색공간·파일 규격까지 로컬에서 정량 측정합니다.")
        with c2:
            st.markdown("**② 과보정 방지 보정**")
            st.write("스톡 전용 프리셋 5종. 채도 상한·할로 방지 샤프닝·하이라이트 보호로 "
                     "디자이너가 2차 편집하기 좋은 상태를 만듭니다.")
        with c3:
            st.markdown("**③ 메타데이터 & CSV**")
            st.write("Gemini 키가 있으면 영문 제목·키워드 25~30개·카테고리·IP 리스크를 자동 생성하고, "
                     "Adobe Stock 업로드용 CSV 로 바로 내보냅니다.")
    else:
        data = up.getvalue()
        name = up.name
        h = digest(data)

        with st.spinner("심사 사전 진단 중..."):
            ana = cached_analyze(data, name, to_srgb)
        img = open_display(data, to_srgb)
        metrics = ana["metrics"]
        report = ana["report"]
        info = ana["info"]

        # ---- 요약 카드 ----
        st.markdown(f"### 🔍 `{name}` 진단 결과")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("심사 준비 점수", f"{report['score']}/100",
                  delta={"pass": "통과권", "caution": "거의 통과", "warn": "손볼 곳 있음", "fail": "반려 위험"}[report["verdict"]],
                  delta_color="normal" if report["verdict"] in ("pass", "caution") else "inverse")
        k2.metric("해상도", f"{metrics['megapixels']:.1f} MP", f"{metrics['width']}×{metrics['height']} {metrics['aspect_label']}")
        k3.metric("하이라이트 날아감", f"{metrics['highlight_clip_pct']:.2f}%", "권장 2% 이하",
                  delta_color="off")
        k4.metric("암부 뭉개짐", f"{metrics['shadow_clip_pct']:.2f}%", "권장 8% 이하", delta_color="off")
        k5.metric("선명도 / 깨끗함", f"{metrics['sharpness_score']} / {metrics['noise_score']}", "100점 만점", delta_color="off")

        color = VERDICT_COLOR[report["verdict"]]
        st.markdown(
            f"<div style='padding:10px 14px;border-left:6px solid {color};background:{color}14;border-radius:6px'>"
            f"<b style='color:{color}'>{report['verdict_ko']}</b> — "
            f"통과 {report['counts']['pass']} · 주의 {report['counts']['warn']} · 위험 {report['counts']['fail']}"
            f"</div>", unsafe_allow_html=True)

        with st.expander("📋 점검 항목 전체 보기 (스톡 심사 체크리스트)", expanded=report["counts"]["fail"] > 0):
            df_block(check_table(report), hide_index=True)
            st.caption(f"파일 형식 {metrics['format']} · 색공간 {metrics['icc_name'] or 'sRGB(프로필 없음)'} · "
                       f"용량 {metrics['file_size_mb']:.1f} MB" if metrics["file_size_mb"] else "")

        st.markdown("---")

        # ---- 보정 ----
        params = resolve_params(metrics)

        with st.spinner("보정 중..."):
            res = cached_enhance(data, tuple(sorted((k, round(v, 5)) for k, v in params.items())),
                                 quality, subsampling, to_srgb, keep_metadata, strip_location)
        out_img = Image.open(io.BytesIO(res["jpeg"]))
        post = res["stats"]

        st.markdown(f"### 🔧 보정 결과 — **{se.PRESETS[preset_key]['label']}** "
                    f"{'(자동 튜닝)' if auto_tune and params_key == tuple(sorted((k, round(v, 5)) for k, v in se.preset_params(preset_key).items())) else '(수동 설정)'}")
        c1, c2 = st.columns(2)
        with c1:
            img_block(img, "보정 전 (Before)")
        with c2:
            img_block(out_img, "보정 후 (After)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("하이라이트", f"{post['highlight_clip_pct']:.2f}%",
                  f"{post['highlight_clip_pct'] - metrics['highlight_clip_pct']:+.2f}%p", delta_color="inverse")
        m2.metric("암부", f"{post['shadow_clip_pct']:.2f}%")
        m3.metric("평균 채도", f"{post['saturation_mean']:.3f}")
        m4.metric("출력 크기", f"{post['size'][0]}×{post['size'][1]}")

        with st.expander("🔬 1:1 원본 크기 부분 비교 (디테일·할로 확인)", expanded=False):
            zc1, zc2 = st.columns([1, 1])
            region = zc1.selectbox("확대 위치", ["중앙", "좌상단", "우상단", "좌하단", "우하단"], index=0)
            crop_side = zc2.slider("크롭 크기(px)", 400, 1200, 700, 50)
            w, h = img.size
            side = min(crop_side, w, h)
            xs = {"중앙": (w - side) // 2, "좌상단": 0, "우상단": w - side, "좌하단": 0, "우하단": w - side}[region]
            ys = {"중앙": (h - side) // 2, "좌상단": 0, "우상단": 0, "좌하단": h - side, "우하단": h - side}[region]
            za, zb = st.columns(2)
            za.caption("Before 1:1")
            za.image(img.crop((xs, ys, xs + side, ys + side)))
            zb.caption("After 1:1")
            zb.image(out_img.crop((xs, ys, xs + side, ys + side)))

        with st.expander("🎛️ 실제 적용된 파라미터", expanded=False):
            df_block([{"파라미터": k, "값": v} for k, v in res["params"].items()], hide_index=True)

        # ---- 다운로드 ----
        dl1, dl2 = st.columns(2)
        out_name = se.stock_filename(name)
        dl1.download_button("⬇️ 보정본 JPEG (Q%d, 4:4:4)" % quality, data=res["jpeg"],
                            file_name=out_name, mime="image/jpeg", type="primary")
        dl2.download_button("⬇️ 진단 리포트 (Markdown)", data=report_markdown(name, metrics, report, res["params"], post),
                            file_name=f"{os.path.splitext(name)[0]}_report.md", mime="text/markdown")
        st.caption(f"출력: {out_name} · {human_mb(len(res['jpeg']))} · 품질 {quality} · subsampling {subsampling}")

        st.markdown("---")

        # ---- Gemini 리스크 & 메타데이터 ----
        st.markdown("### 🤖 심사 리스크 점검 · 스톡 메타데이터 (Gemini)")
        if not api_key:
            st.info("사이드바에 **Gemini API 키**를 넣으면 로고·상표·차량 번호판·초상권 리스크 점검과 "
                    "영문 제목·키워드 25~30개·카테고리 자동 생성, **Adobe Stock CSV** 내보내기가 열립니다. "
                    "키가 없으면 아래에서 직접 입력해 CSV 를 만들 수 있습니다.")
        else:
            if st.button("🤖 AI 심사 리스크 + 메타데이터 생성", type="primary"):
                with st.spinner("Gemini 가 사진을 심사하고 메타데이터를 작성 중..."):
                    gres = gs.analyze_image(out_img, api_key, model=model, local_metrics=metrics)
                st.session_state[f"gem_{h}"] = gres

        state_key = f"gem_{h}"
        gres = st.session_state.get(state_key)

        if gres and gres.get("ok"):
            lvl = {"low": "낮음", "medium": "보통", "high": "높음"}.get(gres["risk_level"], gres["risk_level"])
            mark = {"low": "🟢", "medium": "🟠", "high": "🔴"}.get(gres["risk_level"], "⚪")
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.metric("IP 리스크", f"{mark} {lvl}")
                st.metric("릴리즈 필요", "필요" if gres["release_needed"] else "불필요")
                st.metric("에디토리얼 전용", "권장" if gres["editorial_use"] else "아니오")
            with rc2:
                if gres["risk_items"]:
                    df_block([{"유형": i["type"], "심각도": i["severity"], "위치": i["where"] or "-",
                               "조치": i["advice"] or "-"} for i in gres["risk_items"]], hide_index=True)
                else:
                    st.success("탐지된 로고·상표·초상권 리스크 요소가 없습니다.")
                st.write(gres["summary_ko"])
                if gres["quality_notes_ko"]:
                    st.caption(gres["quality_notes_ko"])

            st.markdown("#### 📝 메타데이터 (수정 가능)")
            title = st.text_area("Title (영문, 200자 이내)", value=gres["title_en"], height=70)
            st.caption(f"한국어 참고: {gres['title_ko']}" if gres.get("title_ko") else "")
            kw_default = ", ".join(gres["keywords_en"])
            kw_text = st.text_area(f"Keywords (영문 {len(gres['keywords_en'])}개 · 쉼표 구분)", value=kw_default, height=90)
            cats = [n for _num, n in se.ADOBE_CATEGORIES]
            cat_index = cats.index(gres["category"]) if gres["category"] in cats else 0
            category = st.selectbox("Adobe Stock 카테고리", cats, index=cat_index)

            keywords = [k.strip() for k in re.split(r"[,;]", kw_text) if k.strip()]
            if len(keywords) < 25:
                st.warning(f"키워드 {len(keywords)}개 — Adobe Stock 은 25~30개를 권장합니다.")
            elif len(keywords) > 30:
                st.warning(f"키워드 {len(keywords)}개 — 30개를 넘으면 CSV 에서 앞 30개만 사용하는 것을 권장합니다.")
            else:
                st.success(f"키워드 {len(keywords)}개 — 권장 범위입니다.")

            row = [{"filename": out_name, "title": title, "keywords": keywords[:30], "category": category}]
            cc1, cc2 = st.columns(2)
            cc1.download_button("⬇️ Adobe Stock CSV", se.build_adobe_csv(row),
                                file_name=f"{os.path.splitext(out_name)[0]}_adobe.csv",
                                mime="text/csv", type="primary")
            cc2.download_button("⬇️ Shutterstock CSV (참고)", se.build_shutterstock_csv(row),
                                file_name=f"{os.path.splitext(out_name)[0]}_shutterstock.csv", mime="text/csv")
            with st.expander("CSV 미리보기"):
                st.code(se.build_adobe_csv(row), language="text")

        elif gres and not gres.get("ok"):
            st.error(gres.get("error", "AI 분석에 실패했습니다."))
            st.caption("키 없이도 CSV 는 아래에서 직접 만들 수 있습니다.")

        if not (gres and gres.get("ok")):
            with st.expander("✍️ 메타데이터 직접 입력 → Adobe Stock CSV", expanded=False):
                title = st.text_input("Title (영문)", value="")
                kw_text = st.text_input("Keywords (영문, 쉼표 구분)", value="")
                cats = [n for _num, n in se.ADOBE_CATEGORIES]
                category = st.selectbox("카테고리", cats, index=cats.index("Landscapes") if "Landscapes" in cats else 0)
                keywords = [k.strip() for k in re.split(r"[,;]", kw_text) if k.strip()]
                row = [{"filename": out_name, "title": title, "keywords": keywords, "category": category}]
                st.download_button("⬇️ Adobe Stock CSV", se.build_adobe_csv(row),
                                   file_name=f"{os.path.splitext(out_name)[0]}_adobe.csv", mime="text/csv")


# ===========================================================================
# TAB 2 — 일괄 보정
# ===========================================================================
with tab_batch:
    st.markdown("여러 장을 한 번에 진단·보정합니다. 진단 점수표와 함께 **보정본 ZIP** 과 "
                "**Adobe Stock CSV** 를 한꺼번에 받을 수 있습니다.")

    src_mode = st.radio("사진 가져오기", ["파일 선택", "폴더 경로 (내 PC)"], horizontal=True, key="batch_mode")
    batch_paths: List[str] = []
    uploaded_files = []

    if src_mode == "파일 선택":
        uploaded_files = st.file_uploader(
            "사진들을 한꺼번에 선택하세요 (복수 선택 가능)",
            type=["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"],
            accept_multiple_files=True, key="batch_upload")
        if uploaded_files:
            st.write(f"선택된 사진: **{len(uploaded_files)}장**")
    else:
        fc1, fc2 = st.columns([3, 1])
        folder = fc1.text_input("폴더 경로", placeholder=r"예: C:\Users\me\Pictures\stock_candidates",
                                key="batch_folder")
        recursive = fc2.checkbox("하위 폴더 포함", value=False)
        if folder:
            batch_paths = se.iter_image_files(folder, recursive=recursive)
            if batch_paths:
                st.success(f"이미지 {len(batch_paths)}장을 찾았습니다.")
                with st.expander("파일 목록"):
                    st.write("\n".join(os.path.basename(p) for p in batch_paths[:200]))
            else:
                st.warning("폴더에서 이미지 파일을 찾지 못했습니다. 경로를 확인하세요. "
                           "(이 기능은 앱을 내 PC에서 실행할 때만 동작합니다)")

    n_items = len(uploaded_files) if uploaded_files else len(batch_paths)
    b1, b2 = st.columns([2, 2])
    with b1:
        batch_auto = st.checkbox("사진별 자동 튜닝", value=True, key="batch_auto",
                                 help="각 사진의 클리핑·노출·선명도에 맞춰 프리셋을 자동 보정합니다.")
        save_to_folder = st.checkbox("보정본을 같은 폴더의 stock_output 에도 저장", value=False,
                                     disabled=src_mode != "폴더 경로")
    with b2:
        use_gemini_batch = st.checkbox("AI 메타데이터도 함께 생성 (사진당 1회 호출)", value=False,
                                       disabled=not bool(api_key))
        if not api_key:
            st.caption("Gemini 키를 넣으면 일괄 메타데이터 생성이 가능합니다.")

    run_batch = st.button(f"🚀 {n_items}장 일괄 진단·보정 시작", type="primary", disabled=n_items == 0)

    if run_batch and n_items:
        items: List[Tuple[str, bytes]] = []
        if uploaded_files:
            for f in uploaded_files:
                items.append((f.name, f.getvalue()))
        else:
            for p in batch_paths:
                try:
                    with open(p, "rb") as fh:
                        items.append((os.path.basename(p), fh.read()))
                except OSError as e:
                    st.warning(f"읽기 실패: {p} ({e})")

        progress = st.progress(0.0)
        status = st.empty()
        zip_buf = io.BytesIO()
        rows: List[Dict[str, Any]] = []
        csv_rows: List[Dict[str, Any]] = []
        params_for_batch = current_params if not batch_auto else None
        results: List[Dict[str, Any]] = []
        out_dir = os.path.join(folder, "stock_output") if (save_to_folder and src_mode == "폴더 경로") else None

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, (fname, raw) in enumerate(items, start=1):
                status.text(f"처리 중 ({idx}/{len(items)}): {fname}")
                try:
                    ana = cached_analyze(raw, fname, to_srgb)
                    metrics = ana["metrics"]
                    report = ana["report"]
                    params = params_for_batch or se.auto_tune(metrics, preset_key, guard=True)
                    res = cached_enhance(raw, tuple(sorted((k, round(v, 5)) for k, v in params.items())),
                                         quality, subsampling, to_srgb, keep_metadata, strip_location)
                    out_name = se.stock_filename(fname)
                    zf.writestr(out_name, res["jpeg"])
                    if out_dir:
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                            with open(os.path.join(out_dir, out_name), "wb") as fh:
                                fh.write(res["jpeg"])
                        except OSError:
                            pass

                    issues = [c["label"] for c in report["checks"] if c["severity"] in ("fail", "warn")]
                    rows.append({
                        "파일": fname,
                        "MP": round(metrics["megapixels"], 2),
                        "크기": f"{metrics['width']}×{metrics['height']}",
                        "점수": report["score"],
                        "판정": report["verdict_ko"],
                        "하이라이트%": metrics["highlight_clip_pct"],
                        "암부%": metrics["shadow_clip_pct"],
                        "선명도": metrics["sharpness_score"],
                        "노이즈": metrics["noise_score"],
                        "주의/위험": ", ".join(issues) if issues else "-",
                        "출력파일": out_name,
                    })
                    results.append({"filename": fname, "out_name": out_name, "metrics": metrics,
                                    "report": report, "params": res["params"], "size": len(res["jpeg"]),
                                    "jpeg": res["jpeg"]})

                    csv_row = {"filename": out_name, "title": "", "keywords": [], "category": ""}
                    if use_gemini_batch and api_key:
                        try:
                            gres = gs.analyze_image(Image.open(io.BytesIO(res["jpeg"])), api_key,
                                                    model=model, local_metrics=metrics)
                            if gres.get("ok"):
                                csv_row.update({"title": gres["title_en"],
                                                "keywords": gres["keywords_en"][:30],
                                                "category": gres["category"]})
                                rows[-1]["제목(AI)"] = gres["title_en"][:60]
                            else:
                                rows[-1]["제목(AI)"] = f"실패: {gres.get('error', '')[:40]}"
                            time.sleep(0.6)
                        except Exception as e:  # noqa: BLE001
                            rows[-1]["제목(AI)"] = f"오류: {e}"
                    csv_rows.append(csv_row)
                except Exception as e:  # noqa: BLE001
                    rows.append({"파일": fname, "MP": 0, "크기": "-", "점수": 0, "판정": f"실패: {e}",
                                 "하이라이트%": 0, "암부%": 0, "선명도": 0, "노이즈": 0,
                                 "주의/위험": "-", "출력파일": "-"})
                progress.progress(idx / len(items))

        status.text("완료")
        st.success(f"{len(items)}장 처리 완료 · 보정본 {len(results)}장")
        df_block(rows, hide_index=True)

        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ 보정본 ZIP", zip_buf.getvalue(),
                           file_name=f"stock_batch_{stamp}.zip", mime="application/zip", type="primary")
        d2.download_button("⬇️ Adobe Stock CSV", se.build_adobe_csv(csv_rows),
                           file_name=f"stock_batch_{stamp}_adobe.csv", mime="text/csv",
                           disabled=not use_gemini_batch)
        rep_buf = io.StringIO()
        w = csv.writer(rep_buf)
        if rows:
            w.writerow(list(rows[0].keys()))
            for r in rows:
                w.writerow(list(r.values()))
        d3.download_button("⬇️ 진단 리포트 CSV", rep_buf.getvalue(),
                           file_name=f"stock_batch_{stamp}_report.csv", mime="text/csv")

        if use_gemini_batch and csv_rows and any(r["title"] for r in csv_rows):
            with st.expander("Adobe Stock CSV 미리보기"):
                st.code(se.build_adobe_csv(csv_rows), language="text")
        elif use_gemini_batch:
            st.caption("AI 메타데이터가 생성된 사진이 없어 CSV 가 비어 있습니다.")

        with st.expander("개별 결과 다시 보기"):
            for r in results:
                st.markdown(f"**{r['filename']}** → `{r['out_name']}` · {r['report']['score']}점 · "
                            f"{r['report']['verdict_ko']} · {human_mb(r['size'])}")
                st.image(r["jpeg"])


# ===========================================================================
# TAB 3 — 기준 / 사용법
# ===========================================================================
with tab_help:
    st.markdown("## 📚 Adobe Stock 사진 심사 기준 요약")
    st.markdown(
        """
| 항목 | 기준 | 이 앱의 처리 |
|---|---|---|
| 해상도 | **최소 4 MP** (권장 6MP↑) | 4MP 미만이면 즉시 경고, 업스케일 대신 원본 사용 권고 |
| 파일 형식 | 사진은 JPEG | PNG/TIFF 입력도 고품질 JPEG(Q95~98, 4:4:4)로 변환 저장 |
| 파일 용량 | 최대 100 MB | 초과 시 경고 |
| 색공간 | sRGB 권장 | 비 sRGB 프로필은 원클릭 변환 |
| 노출 | 과노출/노출부족 반려 | 하이라이트/암부 클리핑 % 측정 후 자동 보호 |
| 화질 | 노이즈·흐림 반려 | Laplacian 선명도 + 노이즈 추정치로 사전 차단 |
| 가공 | 과도한 보정·필터 반려 | 채도 상한·할로 방지 샤프닝·클리핑 없는 S커브 |
| 권리 | 로고·상표·차량 번호판·초상권 | Gemini Vision 리스크 점검 + 릴리즈 필요 여부 표시 |
| 메타데이터 | 영문 제목 + 키워드 25~30개 + 카테고리 | 자동 생성 후 수정 가능, CSV 내보내기 |
        """
    )

    st.markdown("## 🎨 프리셋 5종 가이드")
    for key in se.PRESET_ORDER:
        p = se.PRESETS[key]
        with st.expander(f"**{p['label']}** — {p['label_ko']}", expanded=False):
            st.write(p["desc"])
            df_block([{"파라미터": k, "값": round(float(p[k]), 4)} for k in se.PARAM_RANGES], hide_index=True)

    st.markdown("## 📐 측정 지표 읽는 법")
    st.markdown(
        """
- **하이라이트 날아감(%)**: 휘도 250 이상 픽셀 비율. 2% 이하면 안전, 6% 이상이면 '과노출' 반려 위험.
- **암부 뭉개짐(%)**: 휘도 4 이하 픽셀 비율. 8% 이하면 안전, 20% 이상이면 '노출 부족' 위험.
- **선명도(Laplacian σ)**: 장변 1600px 로 정규화한 뒤 라플라시안 표준편차를 측정. 값이 클수록 선명.
  해상도에 따라 값이 달라지지 않도록 정규화했고, 0~100 점수로 환산합니다.
- **노이즈 σ**: 평탄한 영역(하위 15% 그라디언트)의 고주파 잔차 중위절대편차(MAD). 질감과 노이즈를 구분합니다.
- **과채도 픽셀(%)**: 채도 0.92 초과 비율. 3% 이상이면 과보정으로 보일 수 있습니다.
- **심사 준비 점수**: 위험 1건당 −18, 주의 1건당 −7. 100점에서 차감.
        """
    )

    st.markdown("## ❓ 자주 묻는 질문")
    with st.expander("해상도가 4MP 미만이면 업스케일해서 내면 되나요?"):
        st.write("권장하지 않습니다. Adobe Stock 은 업스케일 흔적(보간 아티팩트)을 화질 저하로 판정할 수 있습니다. "
                 "원본 해상도로 다시 현상하거나, 다른 사진을 선택하는 편이 안전합니다.")
    with st.expander("보정을 얼마나 해도 되나요?"):
        st.write("스톡은 '디자이너가 이어서 편집할 수 있는 상태'를 좋아합니다. "
                 "이 앱은 채도 상한·할로 방지·하이라이트 보호로 그 선을 넘지 않게 설계되어 있습니다. "
                 "과한 필름 룩·비네팅·HDR 느낌은 반려 사유입니다.")
    with st.expander("CSV 는 어떻게 쓰나요?"):
        st.write("Adobe Stock 웹 → **Contributor → Upload → Bulk upload** 에서 CSV 를 함께 올리면 "
                 "파일명 기준으로 제목·키워드·카테고리가 자동 입력됩니다. "
                 "헤더는 `Filename,Title,Keywords,Category,Releases` 이고, 카테고리는 1~21 숫자입니다.")
    with st.expander("Gemini API 키는 꼭 필요한가요?"):
        st.write("아니요. 진단·보정·일괄처리·ZIP·CSV(수동 입력) 는 키 없이 100% 동작합니다. "
                 "키를 넣으면 ① 로고·상표·번호판·초상권 리스크 점검 ② 영문 제목·키워드 25~30개·카테고리 자동 생성 "
                 "③ 일괄 CSV 자동 채우기가 추가됩니다.")

    st.markdown("## 🚀 실행 방법 (README 참고)")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.caption("Windows 는 run_windows.bat 더블클릭으로 실행할 수 있습니다.")

st.markdown("---")
st.caption("📸 스톡사진 심사 통과 자동보정 스튜디오 · 로컬 처리(사진은 외부로 전송되지 않습니다 — "
           "Gemini 를 켠 경우에만 해당 사진 1장이 Google API 로 전송됩니다)")
