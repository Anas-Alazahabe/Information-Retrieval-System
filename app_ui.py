import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib
import inspect

import requests
import streamlit as st

import shared.ir_config as ir_config
import ui.health as health_module

_REQUIRED_IR_CONFIG_ATTRS = (
    "PERSONALIZATION_URL",
    "CLUSTERING_URL",
    "RAG_URL",
    "clustering_health_url",
    "clustering_meta_url",
    "clustering_comparison_url",
    "rag_health_url",
)
if any(not hasattr(ir_config, attr) for attr in _REQUIRED_IR_CONFIG_ATTRS):
    ir_config = importlib.reload(ir_config)

if "clustering_health_url" not in inspect.signature(
    health_module.fetch_system_status_cached
).parameters:
    health_module = importlib.reload(health_module)

fetch_system_status_cached = health_module.fetch_system_status_cached
render_system_status = health_module.render_system_status

HISTORY_MAX_QUERIES = ir_config.HISTORY_MAX_QUERIES
PERSONALIZATION_URL = ir_config.PERSONALIZATION_URL
CLUSTERING_URL = ir_config.CLUSTERING_URL
RAG_URL = ir_config.RAG_URL
REFINEMENT_URL = ir_config.REFINEMENT_URL
RETRIEVAL_URL = ir_config.RETRIEVAL_URL
suggest_url = ir_config.suggest_url

from shared.search_pipeline import search_with_rag
from ui.clustering import render_clustering_section
from ui.rag_answer import render_rag_answer
from ui.results import render_results
from ui.search_input import render_search_input
from ui.sidebar import render_sidebar
from ui.styles import inject_styles
from ui.transparency import render_summary_strip, render_technical_details

st.set_page_config(
    page_title="محرك البحث الذكي - نظام استرجاع المعلومات 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEALTH_CHECK_URL = f"{RETRIEVAL_URL.rstrip('/')}/health"
REFINEMENT_HEALTH_URL = f"{REFINEMENT_URL.rstrip('/')}/health"
PERSONALIZATION_HEALTH_URL = f"{PERSONALIZATION_URL.rstrip('/')}/health"
CLUSTERING_HEALTH_URL = ir_config.clustering_health_url()
CLUSTERING_META_URL = ir_config.clustering_meta_url()
CLUSTERING_COMPARISON_URL = ir_config.clustering_comparison_url()
RAG_HEALTH_URL = ir_config.rag_health_url()
SUGGEST_SERVICE_URL = suggest_url()

_SESSION_DEFAULTS = {
    "last_search_results": {},
    "last_search_query": "",
    "last_search_data": None,
    "last_refinement_meta": None,
    "last_personalization_meta": None,
    "last_rag_meta": None,
    "last_use_personalization": False,
    "last_use_rag": False,
    "last_use_refinement": False,
    "last_personalization_user_id": None,
    "query": "",
    "search_history": [],
    "trigger_search": False,
    "suggest_url": SUGGEST_SERVICE_URL,
}

for key, default in _SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.get("pending_query"):
    st.session_state.query = st.session_state.pop("pending_query")

inject_styles()

st.title("محرك البحث الأكاديمي المطور")
st.markdown(
    "<p class='welcome-subtitle'>ابحث في المقاطع المفهرسة، حسّن استعلامك، "
    "وخصّص النتائج حسب اهتماماتك.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

system_status = fetch_system_status_cached(
    HEALTH_CHECK_URL,
    REFINEMENT_HEALTH_URL,
    PERSONALIZATION_HEALTH_URL,
    clustering_health_url=CLUSTERING_HEALTH_URL,
    rag_health_url=RAG_HEALTH_URL,
)

settings = render_sidebar(
    personalization_url=PERSONALIZATION_URL,
    personalize_profile_url_fn=ir_config.personalize_profile_url,
    rag_ready=system_status.get("rag_ok", False),
    rag_gemini_configured=system_status.get("rag_gemini", False),
)

render_system_status(
    system_status,
    use_refinement=settings.use_refinement,
    use_personalization=settings.use_personalization,
    use_rag=settings.use_rag,
    refinement_techniques=settings.refinement_techniques,
    search_history_len=len(st.session_state["search_history"]),
)

query, should_search = render_search_input()

if should_search:
    if not query.strip():
        st.warning("اكتب استعلاماً ثم اضغط «ابحث الآن».")
    else:
        with st.spinner("جاري البحث وتحسين الاستعلام وترتيب النتائج..."):
            try:
                previous_queries = []
                if settings.use_refinement and "history" in settings.refinement_techniques:
                    previous_queries = list(st.session_state["search_history"])

                pipeline_result = search_with_rag(
                    raw_query=query,
                    representation_mode=settings.representation_mode,
                    use_refinement=settings.use_refinement,
                    use_personalization=bool(
                        settings.use_personalization and settings.personalization_user_id
                    ),
                    use_rag=settings.use_rag,
                    user_id=settings.personalization_user_id,
                    techniques=settings.refinement_techniques,
                    previous_queries=previous_queries,
                    k1=settings.k1,
                    b=settings.b,
                    top_n_filter=settings.top_n_filter,
                    top_k=settings.display_top_k,
                    k_rrf=settings.k_rrf,
                    bm25_rrf_weight=settings.bm25_rrf_weight,
                    embedding_rrf_weight=settings.embedding_rrf_weight,
                    alpha=settings.personalization_alpha,
                    rag_top_context_docs=settings.rag_top_context_docs,
                )
                data = pipeline_result["search"]
                refinement_meta = pipeline_result["refinement"]
                personalization_meta = pipeline_result["personalization"]
                rag_meta = pipeline_result["rag"]

                if data["status"] == "success":
                    normalized_query = query.strip()
                    if normalized_query:
                        history = st.session_state["search_history"]
                        if not history or history[-1] != normalized_query:
                            history.append(normalized_query)
                        st.session_state["search_history"] = history[-HISTORY_MAX_QUERIES:]

                    st.session_state["last_search_data"] = data
                    st.session_state["last_refinement_meta"] = refinement_meta
                    st.session_state["last_personalization_meta"] = personalization_meta
                    st.session_state["last_rag_meta"] = rag_meta
                    st.session_state["last_search_results"] = data.get("results", {})
                    st.session_state["last_search_query"] = normalized_query
                    st.session_state["last_use_refinement"] = settings.use_refinement
                    st.session_state["last_use_personalization"] = bool(
                        settings.use_personalization and settings.personalization_user_id
                    )
                    st.session_state["last_use_rag"] = settings.use_rag
                    st.session_state["last_personalization_user_id"] = (
                        settings.personalization_user_id
                    )
                else:
                    st.error("تعذّر إكمال البحث — تحقق من حالة الخدمات.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "تعذّر الاتصال بإحدى خدمات البحث. "
                    "تأكد من تشغيل الخدمات ثم أعد المحاولة."
                )
                with st.expander("تفاصيل تقنية للاتصال"):
                    services = "preprocessing (8000)، retrieval (8002)"
                    if settings.use_refinement:
                        services += "، refinement (8003)"
                    if settings.use_personalization and settings.personalization_user_id:
                        services += "، personalization (8004)"
                    if settings.use_rag:
                        services += "، rag (8006)"
                    st.caption(f"الخدمات المطلوبة: {services}")
            except Exception as exc:
                st.error(f"حدث خطأ أثناء البحث: {exc}")

last_data = st.session_state.get("last_search_data")
if last_data and last_data.get("status") == "success":
    last_refinement = st.session_state.get("last_refinement_meta")
    last_personalization = st.session_state.get("last_personalization_meta")
    last_rag = st.session_state.get("last_rag_meta")
    last_query = st.session_state.get("last_search_query", "")
    results = last_data.get("results", {})
    total_results = last_data.get("total_results", len(results))

    st.markdown("---")
    if total_results > 0:
        st.success(f"تم العثور على {total_results} نتيجة")
    else:
        st.info(
            "لم نجد نتائج — جرّب كلمات أبسط أو فعّل البحث المحسّن "
            "أو غيّر طريقة الاسترجاع."
        )

    render_summary_strip(
        last_data,
        last_refinement,
        last_personalization,
        use_refinement=st.session_state.get("last_use_refinement", False),
        use_personalization=st.session_state.get("last_use_personalization", False),
        rag_meta=last_rag,
        use_rag=st.session_state.get("last_use_rag", False),
    )

    if total_results > 0:
        render_rag_answer(
            last_rag,
            use_rag=st.session_state.get("last_use_rag", False),
        )
        render_results(
            results,
            query=last_query,
            refinement_meta=last_refinement,
            personalization_meta=last_personalization,
            use_personalization=st.session_state.get("last_use_personalization", False),
            personalization_user_id=st.session_state.get("last_personalization_user_id"),
        )

    render_technical_details(
        last_data,
        last_refinement,
        last_personalization,
        rag_meta=last_rag,
    )

st.markdown("---")
render_clustering_section(
    CLUSTERING_URL,
    clustering_health_url=CLUSTERING_HEALTH_URL,
    clustering_meta_url=CLUSTERING_META_URL,
    clustering_comparison_url=CLUSTERING_COMPARISON_URL,
    system_status=system_status,
)
