import sys
import time
from pathlib import Path

import requests
import streamlit as st
from st_keyup import st_keyup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.ir_config import (
    HISTORY_MAX_QUERIES,
    REFINEMENT_URL,
    RETRIEVAL_URL,
    SERIAL_HYBRID_TOP_N,
    SUGGEST_DEFAULT_LIMIT,
    SUGGEST_MIN_PREFIX_LEN,
    VALID_REFINEMENT_TECHNIQUES,
    suggest_url,
)
from shared.search_pipeline import search_with_optional_refinement

st.set_page_config(
    page_title="محرك البحث الذكي - نظام استرجاع المعلومات 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEALTH_CHECK_URL = f"{RETRIEVAL_URL.rstrip('/')}/health"
REFINEMENT_HEALTH_URL = f"{REFINEMENT_URL.rstrip('/')}/health"
SUGGEST_SERVICE_URL = suggest_url()
STATUS_CACHE_KEY = "system_status_cache"
STATUS_CACHE_TTL_SEC = 45
SUGGESTION_DISPLAY_MAX_LEN = 60
ENHANCED_QUERY_SUMMARY_MAX_LEN = 80

REFINEMENT_TECHNIQUE_LABELS = {
    "query_preprocess": "معالجة الأسئلة الطبيعية",
    "prf": "توسيع بالملاحظات",
    "synonyms": "توسيع بالمرادفات",
    "history": "سياق البحث السابق",
}

REFINEMENT_PRESETS = {
    "موصى به": ["query_preprocess", "prf", "synonyms"],
    "أسئلة طبيعية": ["query_preprocess"],
    "سجل الجلسة": ["query_preprocess", "history"],
    "تخصيص متقدم": None,
}

SEARCH_MODES = {
    "بحث مباشر": False,
    "بحث محسّن": True,
}

REFINEMENT_TECHNIQUE_OPTIONS = list(VALID_REFINEMENT_TECHNIQUES)

if "query" not in st.session_state:
    st.session_state["query"] = ""
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False

if st.session_state.get("pending_query"):
    st.session_state.query = st.session_state.pop("pending_query")


def _truncate(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _fetch_suggestions(prefix: str) -> list[str]:
    if len(prefix.strip()) < SUGGEST_MIN_PREFIX_LEN:
        return []
    try:
        response = requests.get(
            SUGGEST_SERVICE_URL,
            params={"q": prefix, "limit": SUGGEST_DEFAULT_LIMIT},
            timeout=2,
        )
        if response.status_code == 200:
            return response.json().get("suggestions", [])
    except Exception:
        pass
    return []


def _fetch_system_status() -> dict:
    status = {
        "retrieval_ok": False,
        "index_ready": False,
        "refinement_ok": False,
        "suggestions_loaded": False,
        "suggestions_count": 0,
    }
    try:
        health_resp = requests.get(HEALTH_CHECK_URL, timeout=2)
        if health_resp.status_code == 200:
            status["retrieval_ok"] = True
            status["index_ready"] = bool(health_resp.json().get("index_files_detected"))
    except Exception:
        pass

    try:
        refine_health = requests.get(REFINEMENT_HEALTH_URL, timeout=2)
        if refine_health.status_code == 200:
            status["refinement_ok"] = True
            health_data = refine_health.json()
            status["suggestions_loaded"] = bool(health_data.get("suggestions_index_loaded"))
            status["suggestions_count"] = health_data.get("suggestions_count", 0)
    except Exception:
        pass

    return status


def _fetch_system_status_cached() -> dict:
    cache = st.session_state.get(STATUS_CACHE_KEY)
    now = time.time()
    if cache and (now - cache["fetched_at"]) < STATUS_CACHE_TTL_SEC:
        return cache["status"]
    status = _fetch_system_status()
    st.session_state[STATUS_CACHE_KEY] = {"status": status, "fetched_at": now}
    return status


def _render_enhancement_summary(refinement_meta: dict) -> None:
    raw_query = refinement_meta.get("raw_query", "")
    refined_query = refinement_meta.get("refined_query", "")
    if raw_query and refined_query and raw_query != refined_query:
        st.caption(
            f"تم تحسين الاستعلام: {raw_query} → "
            f"{_truncate(refined_query, ENHANCED_QUERY_SUMMARY_MAX_LEN)}"
        )


def _render_enhancement_details(refinement_meta: dict) -> None:
    col_raw, col_refined = st.columns(2)
    col_raw.markdown(f"**استعلامك:** {refinement_meta.get('raw_query', '')}")
    col_refined.markdown(f"**الاستعلام المحسّن:** {refinement_meta.get('refined_query', '')}")

    if refinement_meta.get("techniques_applied"):
        labels = [
            REFINEMENT_TECHNIQUE_LABELS.get(t, t)
            for t in refinement_meta["techniques_applied"]
        ]
        st.caption(f"التحسينات المطبّقة: {', '.join(labels)}")

    explanation = refinement_meta.get("explanation", "")
    expanded_terms = refinement_meta.get("expanded_terms") or []
    if explanation:
        st.info(explanation)
    elif expanded_terms:
        st.caption(f"مصطلحات مضافة: {', '.join(expanded_terms)}")


def _format_timing(timing: dict) -> str:
    if not timing:
        return ""
    parts = []
    for key, label in (
        ("preprocess_ms", "معالجة"),
        ("encode_ms", "تضمين"),
        ("match_ms", "مطابقة"),
        ("rank_ms", "ترتيب"),
        ("ranking_ms", "ترتيب"),
        ("total_ms", "الإجمالي"),
    ):
        if key in timing:
            parts.append(f"{label}: {timing[key]:.1f} ms")
    return " | ".join(parts)


st.title("🔍 محرك البحث الأكاديمي المطور")
st.markdown("---")

st.sidebar.header("🛠️ إعدادات البحث")

representation_mode = st.sidebar.selectbox(
    "طريقة الاسترجاع:",
    options=[
        ("vsm", "Vector Space Model (TF-IDF)"),
        ("bm25", "Okapi BM25 (النصي الحركي)"),
        ("embedding", "Semantic Embedding (الدلالي العميق)"),
        ("hybrid_parallel", "Hybrid Parallel (الهجين المتوازي عبر RRF)"),
        ("hybrid_serial", "Hybrid Serial (الهجين التسلسلي مع Re-ranking)"),
    ],
    format_func=lambda x: x[1],
)[0]

st.sidebar.markdown("---")
st.sidebar.subheader("تحسين جودة البحث")

search_mode = st.sidebar.radio(
    "تحسين البحث",
    list(SEARCH_MODES.keys()),
    index=0,
    help="بحث مباشر = استعلام كما هو. بحث محسّن = تحسين الاستعلام قبل الاسترجاع.",
)
use_refinement = SEARCH_MODES[search_mode]

refinement_techniques: list[str] = []
if use_refinement:
    preset_name = st.sidebar.selectbox(
        "ملف التحسين",
        options=list(REFINEMENT_PRESETS.keys()),
        index=0,
    )
    preset_techniques = REFINEMENT_PRESETS[preset_name]
    if preset_techniques is None:
        refinement_techniques = st.sidebar.multiselect(
            "تحسينات مخصصة",
            options=REFINEMENT_TECHNIQUE_OPTIONS,
            default=["query_preprocess"],
            format_func=lambda t: REFINEMENT_TECHNIQUE_LABELS.get(t, t),
        )
    else:
        refinement_techniques = preset_techniques

st.sidebar.markdown("---")

k1 = 1.5
b = 0.75

if representation_mode in ["bm25", "hybrid_parallel", "hybrid_serial"]:
    st.sidebar.subheader("🎛️ معاملات BM25")
    k1 = st.sidebar.slider(
        "المعامل k1 (تحجيم تكرار المصطلحات):",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )
    b = st.sidebar.slider(
        "المعامل b (عقوبة طول المستند):",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

top_n_filter = SERIAL_HYBRID_TOP_N
if representation_mode == "hybrid_serial":
    top_n_filter = st.sidebar.slider(
        "عدد مرشحي BM25 لإعادة الترتيب:",
        min_value=10,
        max_value=500,
        value=SERIAL_HYBRID_TOP_N,
        step=10,
    )

display_top_k = st.sidebar.slider(
    "عدد النتائج المعروضة (top_k):",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

system_status = _fetch_system_status_cached()
if system_status["retrieval_ok"] and system_status["index_ready"]:
    if not use_refinement or system_status["refinement_ok"]:
        st.sidebar.success("جاهز للبحث")
    else:
        st.sidebar.warning("البحث المحسّن غير متاح — خدمة التحسين غير متصلة")
elif system_status["retrieval_ok"]:
    st.sidebar.warning("الفهرس غير محمّل")
else:
    st.sidebar.error("الخدمة غير متاحة")

with st.sidebar.expander("تفاصيل النظام"):
    if system_status["retrieval_ok"]:
        st.caption("خدمة الاسترجاع: متصلة")
        st.caption(
            "فهرس البحث: جاهز"
            if system_status["index_ready"]
            else "فهرس البحث: غير موجود"
        )
    else:
        st.caption("خدمة الاسترجاع: غير متصلة")

    if system_status["refinement_ok"]:
        st.caption("خدمة التحسين: متصلة")
        if system_status["suggestions_loaded"]:
            st.caption(
                f"فهرس الاقتراحات: {system_status['suggestions_count']:,} استعلام"
            )
        else:
            st.caption(
                "فهرس الاقتراحات: غير مبني "
                "(شغّل scripts/build_query_suggestion_index.py)"
            )
    else:
        st.caption("خدمة التحسين: غير متصلة (اختيارية للبحث المباشر)")

    if use_refinement and refinement_techniques:
        st.caption(f"التحسينات النشطة: {', '.join(refinement_techniques)}")
    if st.session_state["search_history"]:
        st.caption(
            f"عمليات البحث الأخيرة في الجلسة: {len(st.session_state['search_history'])}"
        )

query = st_keyup(
    "✍️ أدخل استعلامك باللغة الطبيعية هنا:",
    value=st.session_state.get("query", ""),
    key="query",
    debounce=300,
    placeholder="مثال: hospital system أو how to tie a tie...",
) or ""

suggestions = _fetch_suggestions(query)
if query.strip() and len(query.strip()) >= SUGGEST_MIN_PREFIX_LEN and suggestions:
    st.caption("اقتراحات")
    for index, suggestion in enumerate(suggestions):
        label = _truncate(suggestion, SUGGESTION_DISPLAY_MAX_LEN)
        if st.button(label, key=f"suggest_{index}_{hash(suggestion) % 10_000}", help=suggestion):
            st.session_state.pending_query = suggestion
            st.session_state.trigger_search = True
            st.rerun()

if st.button("🚀 ابحث الآن", type="primary"):
    st.session_state.trigger_search = True

should_search = st.session_state.pop("trigger_search", False)
if should_search:
    if not query.strip():
        st.warning("الرجاء كتابة نص البحث أولاً!")
    else:
        with st.spinner("جاري معالجة الاستعلام، حساب الأوزان والمطابقة عبر الخدمات المصغرة..."):
            try:
                previous_queries = []
                if use_refinement and "history" in refinement_techniques:
                    previous_queries = list(st.session_state["search_history"])

                pipeline_result = search_with_optional_refinement(
                    raw_query=query,
                    representation_mode=representation_mode,
                    use_refinement=use_refinement,
                    techniques=refinement_techniques,
                    previous_queries=previous_queries,
                    k1=k1,
                    b=b,
                    top_n_filter=top_n_filter,
                    top_k=display_top_k,
                )
                data = pipeline_result["search"]
                refinement_meta = pipeline_result["refinement"]

                if data["status"] == "success":
                    normalized_query = query.strip()
                    if normalized_query:
                        history = st.session_state["search_history"]
                        if not history or history[-1] != normalized_query:
                            history.append(normalized_query)
                        st.session_state["search_history"] = history[-HISTORY_MAX_QUERIES:]

                    if refinement_meta:
                        _render_enhancement_summary(refinement_meta)

                    results = data["results"]
                    total_results = data["total_results"]

                    st.success(
                        f"🎯 تم العثور على {total_results} وثيقة مطابقة باستخدام نمط ({data['mode_used']})"
                    )
                    matching_method = data.get("matching_method")
                    if matching_method:
                        st.caption(f"طريقة المطابقة: {matching_method}")
                    params = data.get("params")
                    if params:
                        st.caption(f"معاملات المطابقة: {params}")

                    if total_results == 0:
                        st.info(
                            "⚠️ لم يتم العثور على نتائج مطابقة لهذا الاستعلام في الفهارس الحالية."
                        )
                    else:
                        st.markdown("### 📋 قائمة الوثائق المسترجعة مرتبة تنازلياً حسب الـ Score:")
                        table_data = []
                        for rank, (doc_id, score) in enumerate(results.items(), 1):
                            table_data.append(
                                {
                                    "الترتيب (Rank)": f"#{rank}",
                                    "معرف الوثيقة (Document ID)": doc_id,
                                    "علامة المطابقة (Relevance Score)": score,
                                }
                            )
                        st.table(table_data)

                    with st.expander("تفاصيل البحث", expanded=False):
                        if refinement_meta:
                            _render_enhancement_details(refinement_meta)
                        st.caption(
                            f"الـ Tokens بعد المعالجة: {data.get('query_tokens', [])}"
                        )
                        timing_text = _format_timing(data.get("timing", {}))
                        if timing_text:
                            st.caption(f"زمن التنفيذ: {timing_text}")

            except requests.exceptions.ConnectionError:
                if use_refinement:
                    st.error(
                        "تعذّر الاتصال بإحدى الخدمات. "
                        "شغّل preprocessing (8000)، retrieval (8002)، refinement (8003)."
                    )
                else:
                    st.error("❌ فشل الاتصال بخدمة الاسترجاع. تأكد من تشغيل السيرفر!")
            except Exception as exc:
                st.error(f"⚠️ حدث خطأ أثناء جلب البيانات: {str(exc)}")
