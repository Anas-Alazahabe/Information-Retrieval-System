"""System health checks and human-readable status for the sidebar."""

import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

from ui.constants import STATUS_CACHE_KEY, STATUS_CACHE_TTL_SEC


def fetch_system_status(
    retrieval_health_url: str,
    refinement_health_url: str,
    personalization_health_url: str,
    clustering_health_url: Optional[str] = None,
    rag_health_url: Optional[str] = None,
) -> Dict[str, Any]:
    status = {
        "retrieval_ok": False,
        "index_ready": False,
        "refinement_ok": False,
        "personalization_ok": False,
        "personalization_db": False,
        "clustering_ok": False,
        "cluster_artifacts_ready": False,
        "rag_ok": False,
        "rag_gemini": False,
        "rag_db": False,
        "suggestions_loaded": False,
        "suggestions_count": 0,
    }
    try:
        health_resp = requests.get(retrieval_health_url, timeout=10)
        if health_resp.status_code == 200:
            status["retrieval_ok"] = True
            status["index_ready"] = bool(health_resp.json().get("index_files_detected"))
    except Exception:
        pass

    try:
        refine_health = requests.get(refinement_health_url, timeout=2)
        if refine_health.status_code == 200:
            status["refinement_ok"] = True
            health_data = refine_health.json()
            status["suggestions_loaded"] = bool(health_data.get("suggestions_index_loaded"))
            status["suggestions_count"] = health_data.get("suggestions_count", 0)
    except Exception:
        pass

    try:
        personalization_health = requests.get(personalization_health_url, timeout=2)
        if personalization_health.status_code == 200:
            health_data = personalization_health.json()
            status["personalization_ok"] = True
            status["personalization_db"] = bool(health_data.get("database_connected"))
    except Exception:
        pass

    if clustering_health_url:
        try:
            clustering_health = requests.get(clustering_health_url, timeout=2)
            if clustering_health.status_code == 200:
                health_data = clustering_health.json()
                status["clustering_ok"] = True
                status["cluster_artifacts_ready"] = bool(
                    health_data.get("cluster_artifacts_ready")
                )
        except Exception:
            pass

    if rag_health_url:
        try:
            rag_health = requests.get(rag_health_url, timeout=2)
            if rag_health.status_code == 200:
                health_data = rag_health.json()
                status["rag_ok"] = True
                status["rag_gemini"] = bool(health_data.get("gemini_configured"))
                status["rag_db"] = bool(health_data.get("database_connected"))
        except Exception:
            pass

    return status


def fetch_system_status_cached(
    retrieval_health_url: str,
    refinement_health_url: str,
    personalization_health_url: str,
    clustering_health_url: Optional[str] = None,
    rag_health_url: Optional[str] = None,
) -> Dict[str, Any]:
    cache = st.session_state.get(STATUS_CACHE_KEY)
    now = time.time()
    if cache and (now - cache["fetched_at"]) < STATUS_CACHE_TTL_SEC:
        return cache["status"]
    status = fetch_system_status(
        retrieval_health_url,
        refinement_health_url,
        personalization_health_url,
        clustering_health_url,
        rag_health_url,
    )
    st.session_state[STATUS_CACHE_KEY] = {"status": status, "fetched_at": now}
    return status


def render_system_status(
    system_status: Dict[str, Any],
    *,
    use_refinement: bool,
    use_personalization: bool,
    use_rag: bool = False,
    refinement_techniques: list[str],
    search_history_len: int,
) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**حالة النظام**")
    st.sidebar.caption("ماذا يفعل هذا؟ يخبرك إن كان البحث جاهزاً للاستخدام.")
    st.sidebar.caption("لماذا تستخدمه؟ لتجنب الأخطاء قبل إجراء البحث.")

    if st.sidebar.button("تحديث حالة النظام", key="refresh_system_status"):
        st.session_state.pop(STATUS_CACHE_KEY, None)
        st.rerun()

    if system_status["retrieval_ok"] and system_status["index_ready"]:
        if not use_refinement or system_status["refinement_ok"]:
            st.sidebar.success("البحث جاهز — الفهرس محمّل")
        else:
            st.sidebar.warning(
                "البحث المحسّن غير متاح — استخدم البحث المباشر أو شغّل خدمة التحسين"
            )
    elif system_status["retrieval_ok"]:
        st.sidebar.warning("الفهرس غير محمّل — ابنِ الفهرس أولاً")
    else:
        st.sidebar.error("لا يمكن الاتصال — تحقق من تشغيل الخدمات")

    if system_status["refinement_ok"]:
        st.sidebar.caption("تحسين الاستعلام: متاح")
    elif use_refinement:
        st.sidebar.caption("تحسين الاستعلام: غير متصل")

    if system_status["personalization_ok"] and system_status["personalization_db"]:
        st.sidebar.caption("التخصيص: متاح (قاعدة البيانات متصلة)")
    elif use_personalization:
        st.sidebar.caption("التخصيص: غير متاح — النتائج بدون إعادة ترتيب شخصية")

    if system_status.get("rag_ok") and system_status.get("rag_gemini"):
        db_note = "قاعدة البيانات متصلة" if system_status.get("rag_db") else "قاعدة البيانات غير متصلة"
        st.sidebar.caption(f"RAG: متاح (Gemini + {db_note})")
    elif use_rag:
        st.sidebar.caption("RAG: غير متاح — تحقق من الخدمة ومفتاح Gemini")

    with st.sidebar.expander("تفاصيل تقنية للنظام", expanded=False):
        if system_status["retrieval_ok"]:
            st.caption("retrieval_service (8002): متصلة")
            st.caption(
                "فهرس البحث: جاهز"
                if system_status["index_ready"]
                else "فهرس البحث: غير موجود"
            )
        else:
            st.caption("retrieval_service (8002): غير متصلة")

        if system_status["refinement_ok"]:
            st.caption("query_refinement_service (8003): متصلة")
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
            st.caption("query_refinement_service (8003): غير متصلة (اختيارية)")

        if system_status["personalization_ok"]:
            db_label = "متصل" if system_status["personalization_db"] else "غير متصل"
            st.caption(f"personalization_service (8004): متصلة (MySQL: {db_label})")
        else:
            st.caption("personalization_service (8004): غير متصلة (اختيارية)")

        if system_status.get("clustering_ok"):
            cluster_label = (
                "جاهز"
                if system_status.get("cluster_artifacts_ready")
                else "يحتاج precompute"
            )
            st.caption(f"clustering_service (8005): متصلة ({cluster_label})")
        else:
            st.caption("clustering_service (8005): غير متصلة (اختيارية)")

        if system_status.get("rag_ok"):
            gemini_label = "مضبوط" if system_status.get("rag_gemini") else "غير مضبوط"
            db_label = "متصل" if system_status.get("rag_db") else "غير متصل"
            st.caption(f"rag_service (8006): متصلة (Gemini: {gemini_label}, MySQL: {db_label})")
        else:
            st.caption("rag_service (8006): غير متصلة (اختيارية)")

        st.caption("preprocessing_service (8000): مطلوبة للفهرسة والبحث")

        if use_refinement and refinement_techniques:
            st.caption(f"التحسينات النشطة: {', '.join(refinement_techniques)}")
        if search_history_len:
            st.caption(f"عمليات البحث الأخيرة في الجلسة: {search_history_len}")
