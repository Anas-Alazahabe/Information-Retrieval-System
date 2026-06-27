"""Custom CSS for the Streamlit search UI."""

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
<style>
.section-purpose, .section-benefit {
    font-size: 0.85rem;
    color: #5c6370;
    line-height: 1.45;
}
.section-benefit { margin-bottom: 0.35rem; display: inline-block; }
.summary-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0 1rem 0;
}
.summary-chip {
    background: #f0f4f8;
    border: 1px solid #d8dee6;
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    color: #1f2937;
}
.summary-chip strong { color: #0f4c81; }
.result-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.85rem;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.result-card-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.rank-badge {
    background: #0f4c81;
    color: white;
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-weight: 600;
    font-size: 0.9rem;
}
.personalized-badge {
    background: #e8f5e9;
    color: #1b5e20;
    border: 1px solid #a5d6a7;
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
}
.doc-id {
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    color: #64748b;
}
.score-badge {
    font-size: 0.82rem;
    color: #334155;
    font-weight: 500;
}
.snippet-text {
    font-size: 0.95rem;
    line-height: 1.55;
    color: #1e293b;
    margin: 0.5rem 0;
}
.snippet-text mark {
    background: #fff3bf;
    padding: 0 2px;
    border-radius: 2px;
}
.matched-terms {
    font-size: 0.8rem;
    color: #2e7d32;
    margin-top: 0.35rem;
}
.welcome-subtitle {
    font-size: 1.05rem;
    color: #475569;
    margin-top: -0.5rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )
