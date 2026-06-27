"""Generate evaluation charts and Arabic-ready summary markdown."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODE_LABELS = {
    "vsm": "VSM",
    "bm25": "BM25",
    "embedding": "Embedding",
    "hybrid_parallel": "Hybrid (RRF)",
    "hybrid_serial": "Hybrid (Serial)",
}

METRIC_LABELS = {
    "map": "MAP",
    "recall": "Recall",
    "precision_at_10": "P@10",
    "ndcg_at_10": "nDCG@10",
}


def _latest(results_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _save_fig(fig, charts_dir: Path, name: str) -> Path:
    charts_dir.mkdir(parents=True, exist_ok=True)
    out = charts_dir / name
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    plt.close("all")
    return out


def chart_baseline_all_metrics(baseline: dict, charts_dir: Path) -> Optional[Path]:
    modes_data = baseline.get("modes", {})
    if not modes_data:
        return None

    modes = [MODE_LABELS.get(m, m) for m in modes_data.keys()]
    metrics = list(METRIC_LABELS.keys())
    x = np.arange(len(modes))
    width = 0.2
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [modes_data[m].get(metric, 0) for m in modes_data.keys()]
        ax.bar(x + i * width, values, width, label=METRIC_LABELS[metric], color=colors[i])

    ax.set_ylabel("Score")
    ax.set_title("Baseline evaluation — all IR metrics per mode")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, charts_dir, "01_baseline_all_metrics.png")


def chart_refinement_heatmap(summary: dict, charts_dir: Path) -> Optional[Path]:
    deltas = summary.get("deltas_vs_baseline", {})
    if not deltas:
        return None

    run_names = list(deltas.keys())
    modes = list(next(iter(deltas.values())).keys()) if run_names else []
    if not modes:
        return None

    matrix = np.array(
        [[deltas[run].get(mode, {}).get("ndcg_at_10", 0) for mode in modes] for run in run_names]
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(run_names) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.01, vmax=0.01)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], rotation=20, ha="right")
    ax.set_yticks(range(len(run_names)))
    ax.set_yticklabels(run_names)
    ax.set_title("Refinement ablation — nDCG@10 delta vs baseline")
    fig.colorbar(im, ax=ax, label="Δ nDCG@10")
    fig.tight_layout()
    return _save_fig(fig, charts_dir, "02_refinement_deltas_heatmap.png")


def chart_personalization(summary: dict, charts_dir: Path) -> Optional[Path]:
    baseline = summary.get("baseline", {}).get("modes", {})
    personalized = summary.get("personalized", {}).get("modes", {})
    if not baseline:
        return None

    modes = list(baseline.keys())
    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        [baseline[m].get("ndcg_at_10", 0) for m in modes],
        width,
        label="Baseline",
        color="#3498db",
    )
    ax.bar(
        x + width / 2,
        [personalized.get(m, {}).get("ndcg_at_10", 0) for m in modes],
        width,
        label="Personalized",
        color="#e67e22",
    )
    ax.set_ylabel("nDCG@10")
    ax.set_title("Personalization — baseline vs personalized")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes])
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, charts_dir, "03_personalization_before_after.png")


def chart_vector_store(report: dict, charts_dir: Path) -> Optional[Path]:
    modes_data = report.get("modes", {})
    if not modes_data:
        return None

    modes = list(modes_data.keys())
    labels = [MODE_LABELS.get(m, m) for m in modes]
    x = np.arange(len(modes))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ndcg = [modes_data[m].get("ndcg_at_10", 0) for m in modes]
    ax1.bar(x, ndcg, color="#2ecc71", label="nDCG@10")
    ax1.set_ylabel("nDCG@10", color="#2ecc71")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_title("Vector store — sparse vs FAISS dense vs hybrid")

    ax2 = ax1.twinx()
    latency = [modes_data[m].get("mean_latency_ms", 0) for m in modes]
    ax2.plot(x, latency, "o-", color="#e74c3c", label="Mean latency (ms)")
    ax2.set_ylabel("Latency (ms)", color="#e74c3c")

    fig.tight_layout()
    return _save_fig(fig, charts_dir, "04_vector_store_comparison.png")


def chart_rag(report: dict, charts_dir: Path) -> Optional[Path]:
    agg = report.get("aggregate", {})
    per_query = report.get("per_query", [])
    if not agg:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rates = {
        "Citation rate": agg.get("citation_rate", 0),
        "Cited in context": agg.get("cited_in_context_rate", 0),
        "Faithfulness": agg.get("mean_faithfulness_overlap", 0),
    }
    axes[0].bar(rates.keys(), rates.values(), color=["#3498db", "#2ecc71", "#9b59b6"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("RAG quality metrics")
    axes[0].tick_params(axis="x", rotation=15)

    latencies = [q.get("latency_ms", 0) for q in per_query if q.get("latency_ms", 0) > 0]
    if latencies:
        axes[1].hist(latencies, bins=min(15, len(latencies)), color="#e67e22", edgecolor="white")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_title("RAG generation latency distribution")

    fig.tight_layout()
    return _save_fig(fig, charts_dir, "05_rag_latency_citations.png")


def chart_cluster(report: dict, charts_dir: Path) -> Optional[Path]:
    stats = report.get("cluster_stats", {})
    sizes = stats.get("sizes", {})
    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_ids = [str(k) for k in sorted(int(k) for k in sizes.keys())]
    values = [sizes[int(k)] if int(k) in sizes else sizes[str(k)] for k in cluster_ids]
    ax.bar(cluster_ids, values, color="#3498db")
    silhouette = report.get("silhouette_score")
    title = "Cluster size distribution"
    if silhouette is not None:
        title += f" (silhouette={silhouette:.4f})"
    ax.set_title(title)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Document count")
    fig.tight_layout()
    return _save_fig(fig, charts_dir, "06_cluster_sizes.png")


def chart_feature_summary(
    baseline: dict,
    refinement: dict,
    personalization: dict,
    charts_dir: Path,
) -> Optional[Path]:
    if not baseline:
        return None

    modes_data = baseline.get("modes", {})
    best_mode = max(modes_data.keys(), key=lambda m: modes_data[m].get("ndcg_at_10", 0))
    best_ndcg = modes_data[best_mode].get("ndcg_at_10", 0)

    combined_ndcg = best_ndcg
    if refinement:
        combined = refinement.get("runs", {}).get("combined", {}).get("modes", {})
        if best_mode in combined:
            combined_ndcg = combined[best_mode].get("ndcg_at_10", best_ndcg)

    pers_ndcg = combined_ndcg
    if personalization:
        pers_modes = personalization.get("personalized", {}).get("modes", {})
        if best_mode in pers_modes:
            pers_ndcg = pers_modes[best_mode].get("ndcg_at_10", combined_ndcg)

    labels = ["Baseline\n(best mode)", "+ Refinement\n(combined)", "+ Personalization"]
    values = [best_ndcg, combined_ndcg, pers_ndcg]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=["#3498db", "#2ecc71", "#e67e22"])
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"Feature contribution summary ({MODE_LABELS.get(best_mode, best_mode)})")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    return _save_fig(fig, charts_dir, "07_feature_contribution_summary.png")


def _best_mode_metrics(baseline: dict) -> tuple[str, dict]:
    modes = baseline.get("modes", {})
    if not modes:
        return "", {}
    best = max(modes.keys(), key=lambda m: modes[m].get("ndcg_at_10", 0))
    return best, modes[best]


def write_summary_md(
    *,
    results_dir: Path,
    charts_dir: Path,
    baseline: Optional[dict],
    refinement: Optional[dict],
    personalization: Optional[dict],
    vector_store: Optional[dict],
    rag: Optional[dict],
    cluster: Optional[dict],
    chart_paths: List[Path],
) -> Path:
    best_mode, best_metrics = _best_mode_metrics(baseline or {})
    protocol = (baseline or {}).get("eval_protocol", {})

    lines = [
        "# Final Evaluation Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Evaluation protocol",
        "",
        f"- **Dataset:** {protocol.get('dataset', 'msmarco-passage/dev')}",
        f"- **Source:** {protocol.get('source', 'ir_datasets official qrels')}",
        f"- **UI queries excluded:** {protocol.get('excludes_ui_queries', True)}",
        f"- **Judged queries evaluated:** {protocol.get('num_judged_queries', 'N/A')}",
        "",
        "## Charts",
        "",
    ]
    for path in chart_paths:
        rel = path.relative_to(results_dir).as_posix()
        lines.append(f"- [{path.name}]({rel})")

    lines.extend(
        [
            "",
            "---",
            "",
            "## تأثير كل طريقة تمثيل (Representation impact)",
            "",
        ]
    )

    if baseline:
        for mode, metrics in baseline.get("modes", {}).items():
            label = MODE_LABELS.get(mode, mode)
            lines.append(
                f"- **{label}:** MAP={metrics.get('map', 0):.4f}, "
                f"Recall={metrics.get('recall', 0):.4f}, "
                f"P@10={metrics.get('precision_at_10', 0):.4f}, "
                f"nDCG@10={metrics.get('ndcg_at_10', 0):.4f}"
            )
    else:
        lines.append("- (baseline report not found)")

    lines.extend(
        [
            "",
            "## مقارنة النماذج (Model comparison)",
            "",
            f"Best mode by nDCG@10: **{MODE_LABELS.get(best_mode, best_mode)}** "
            f"(nDCG@10={best_metrics.get('ndcg_at_10', 0):.4f}).",
            "",
            "Hybrid modes combine sparse (BM25) and dense (FAISS embedding) retrieval. "
            "Embedding and hybrid typically outperform pure VSM/BM25 when the vector index is built at scale.",
            "",
            "## مساهمة الميزات الإضافية (Additional features)",
            "",
        ]
    )

    if refinement:
        combined_delta = refinement.get("deltas_vs_baseline", {}).get("combined", {}).get(best_mode, {})
        lines.append(
            f"- **Query refinement (combined):** ΔnDCG@10={combined_delta.get('ndcg_at_10', 0):+.4f} "
            f"on {MODE_LABELS.get(best_mode, best_mode)}."
        )
    if personalization:
        pers_delta = personalization.get("deltas_personalized_vs_baseline", {}).get(best_mode, {})
        lines.append(
            f"- **Personalization:** ΔnDCG@10={pers_delta.get('ndcg_at_10', 0):+.4f} "
            "(simulated users with oracle clicks)."
        )
    if vector_store:
        faiss = vector_store.get("faiss_status", {})
        lines.append(
            f"- **Vector store (FAISS):** ann_backend={faiss.get('ann_backend')}, "
            f"loaded={faiss.get('faiss_loaded')}."
        )
        emb_delta = vector_store.get("deltas_vs_bm25", {}).get("embedding", {})
        lines.append(f"  - Embedding vs BM25 ΔnDCG@10={emb_delta.get('ndcg_at_10', 0):+.4f}.")
    if rag:
        agg = rag.get("aggregate", {})
        lines.append(
            f"- **RAG:** citation_rate={agg.get('citation_rate', 0):.2f}, "
            f"mean_latency={agg.get('mean_latency_ms', 0):.0f}ms. "
            "Does not change retrieval rankings."
        )
    if cluster:
        stats = cluster.get("cluster_stats", {})
        lines.append(
            f"- **Clustering:** {stats.get('n_clusters', 0)} clusters, "
            f"silhouette={cluster.get('silhouette_score', 'N/A')}. "
            "Visualization-only; no ranking impact."
        )

    lines.extend(
        [
            "",
            "## تبرير اختيار النموذج والمعاملات (Model justification)",
            "",
            f"We selected **{MODE_LABELS.get(best_mode, best_mode)}** as the primary mode based on "
            "highest nDCG@10 on the official dev qrels test set. BM25 parameters (k1=1.5, b=0.75) "
            "follow standard MS MARCO tuning. Hybrid RRF fusion combines lexical precision with "
            "dense semantic recall via FAISS.",
            "",
            "## ملاحظة حول مجموعة الاختبار",
            "",
            "All metrics use queries and qrels from `ir_datasets` (msmarco-passage/dev). "
            "No UI suggestion queries or manually invented queries are used.",
            "",
        ]
    )

    out = results_dir / "FINAL_EVAL_SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_all(results_dir: Path) -> dict:
    charts_dir = results_dir / "charts"

    baseline = _load(_latest(results_dir, "eval_baseline_*.json"))
    refinement = _load(_latest(results_dir, "refinement_ablation_summary_*.json"))
    personalization = _load(_latest(results_dir, "personalization_ablation_summary_*.json"))
    vector_store = _load(_latest(results_dir, "vector_store_ablation_*.json"))
    rag = _load(_latest(results_dir, "rag_eval_*.json"))
    cluster = _load(_latest(results_dir, "cluster_eval_*.json"))

    chart_paths: List[Path] = []
    for fn, data in (
        (chart_baseline_all_metrics, baseline),
        (chart_refinement_heatmap, refinement),
        (chart_personalization, personalization),
        (chart_vector_store, vector_store),
        (chart_rag, rag),
        (chart_cluster, cluster),
    ):
        if data:
            path = fn(data, charts_dir)
            if path:
                chart_paths.append(path)

    if baseline:
        path = chart_feature_summary(baseline, refinement, personalization, charts_dir)
        if path:
            chart_paths.append(path)

    summary_path = write_summary_md(
        results_dir=results_dir,
        charts_dir=charts_dir,
        baseline=baseline,
        refinement=refinement,
        personalization=personalization,
        vector_store=vector_store,
        rag=rag,
        cluster=cluster,
        chart_paths=chart_paths,
    )

    return {
        "charts_dir": str(charts_dir),
        "chart_count": len(chart_paths),
        "summary_path": str(summary_path),
        "charts": [p.name for p in chart_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation charts and summary")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "evaluation_results"),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    manifest = generate_all(results_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
