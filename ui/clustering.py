"""Clustering visualization section for the Streamlit UI."""

from typing import Any, Dict, Optional

import requests
import streamlit as st


def render_clustering_section(
    clustering_url: str,
    *,
    clustering_health_url: str,
    clustering_meta_url: str,
    clustering_comparison_url: str,
    system_status: Optional[Dict[str, Any]] = None,
) -> None:
    st.markdown("### تجميع الوثائق (Clustering)")
    st.caption(
        "يعرض توزيع الوثائق في مجموعات دلالية بناءً على تمثيلات الـ Embedding. "
        "يتطلب تشغيل precompute بعد بناء الفهرس."
    )

    status = system_status or {}
    clustering_ok = status.get("clustering_ok", False)
    artifacts_ready = status.get("cluster_artifacts_ready", False)

    if not clustering_ok:
        st.warning(
            "خدمة التجميع غير متصلة — شغّل clustering_service على المنفذ 8005 "
            "(scripts/start_stack.ps1)"
        )
        return

    if not artifacts_ready:
        st.info(
            "ملفات التجميع غير جاهزة. من جذر المشروع نفّذ:\n\n"
            "`python scripts/run_cluster_precompute.py`"
        )
        return

    try:
        meta_resp = requests.get(clustering_meta_url, timeout=10)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
    except Exception as exc:
        st.error(f"تعذّر جلب بيانات التجميع: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("عدد الوثائق", f"{meta.get('document_count', 0):,}")
    col2.metric("عدد المجموعات", meta.get("n_clusters", 0))
    col3.metric("عينة الرسم", meta.get("viz_sample_size", 0))

    if meta.get("embedding_model"):
        st.caption(f"نموذج التضمين: {meta['embedding_model']}")

    try:
        img_resp = requests.get(clustering_comparison_url, timeout=30)
        img_resp.raise_for_status()
        st.image(img_resp.content, caption="تصوّر t-SNE للمجموعات", use_container_width=True)
    except Exception as exc:
        st.error(f"تعذّر تحميل الرسم البياني: {exc}")

    clusters = meta.get("clusters", [])
    if clusters:
        with st.expander("تفاصيل المجموعات", expanded=False):
            for cluster in clusters:
                cid = cluster.get("cluster_id")
                size = cluster.get("size", 0)
                samples = cluster.get("sample_doc_ids", [])
                st.markdown(f"**المجموعة {cid}** — {size:,} وثيقة")
                if samples:
                    st.caption("عينة: " + ", ".join(samples[:5]))
