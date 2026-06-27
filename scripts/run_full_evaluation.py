"""Orchestrate full evaluation: baseline, refinement, personalization, vector store, RAG, clustering, viz."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

# Load .env for GEMINI_API_KEY (gitignored)
_env_path = ROOT / ".env"
if _env_path.is_file():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from evaluation_service.app.main import run_evaluation
from scripts.generate_eval_visualizations import generate_all
from scripts.run_cluster_eval import run_cluster_evaluation
from scripts.run_personalization_eval import run_personalization_evaluation
from scripts.run_rag_eval import run_rag_evaluation
from scripts.run_refinement_ablation import ABLATION_RUNS, _build_summary, _print_table
from scripts.run_vector_store_eval import run_vector_store_evaluation
from shared.ir_config import (
    CLUSTERING_URL,
    EVAL_DATASET_NAME,
    PERSONALIZATION_URL,
    RAG_URL,
    REFINEMENT_URL,
    RETRIEVAL_URL,
    VALID_REPRESENTATION_MODES,
)

SERVICE_CHECKS = [
    ("preprocessing", "http://127.0.0.1:8000/health"),
    ("retrieval", f"{RETRIEVAL_URL.rstrip('/')}/health"),
    ("refinement", f"{REFINEMENT_URL.rstrip('/')}/health"),
    ("personalization", f"{PERSONALIZATION_URL.rstrip('/')}/health"),
    ("clustering", f"{CLUSTERING_URL.rstrip('/')}/health"),
    ("rag", f"{RAG_URL.rstrip('/')}/health"),
]


def _check_services(required: List[str], *, skip_rag: bool) -> dict:
    status = {}
    for name, url in SERVICE_CHECKS:
        if skip_rag and name == "rag":
            status[name] = "skipped"
            continue
        if name not in required and name in ("clustering", "rag", "personalization"):
            continue
        try:
            resp = requests.get(url, timeout=5)
            status[name] = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception as exc:
            status[name] = f"down ({exc.__class__.__name__})"
    return status


def _save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved {path}")
    return path


def run_full_evaluation(
    *,
    dataset: str = EVAL_DATASET_NAME,
    scale: str = "full",
    max_queries: int = 100,
    top_k: int = 10,
    modes: Optional[List[str]] = None,
    output_dir: Path,
    skip_rag: bool = False,
    skip_personalization: bool = False,
    skip_cluster: bool = False,
    rag_max_queries: int = 20,
) -> dict:
    modes = modes or list(VALID_REPRESENTATION_MODES)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: dict = {
        "scale": scale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "max_queries": max_queries,
        "outputs": {},
        "service_status": _check_services(
            ["preprocessing", "retrieval", "refinement", "personalization", "clustering", "rag"],
            skip_rag=skip_rag,
        ),
    }

    # 1. Baseline
    print("\n=== Step 1: Baseline evaluation ===")
    baseline = run_evaluation(
        dataset_name=dataset,
        representation_modes=modes,
        top_k=top_k,
        max_queries=max_queries,
        use_refinement=False,
    )
    baseline["scale"] = scale
    baseline_path = output_dir / f"eval_baseline_{scale}_{timestamp}.json"
    _save_json(baseline_path, baseline)
    manifest["outputs"]["baseline"] = str(baseline_path)

    # 2. Refinement ablation
    print("\n=== Step 2: Refinement ablation ===")
    reports = {}
    for run_name, use_refinement, techniques in ABLATION_RUNS:
        print(f"  Running {run_name} ...")
        report = run_evaluation(
            dataset_name=dataset,
            representation_modes=modes,
            top_k=top_k,
            max_queries=max_queries,
            use_refinement=use_refinement,
            refinement_techniques=techniques,
        )
        report["scale"] = scale
        reports[run_name] = report
        if use_refinement:
            slug = "-".join(techniques)
            fname = f"eval_refined_{slug}_{scale}_{timestamp}.json"
        else:
            fname = f"eval_baseline_{scale}_{timestamp}.json"
        _save_json(output_dir / fname, report)

    refinement_summary = _build_summary(scale, reports)
    refinement_path = output_dir / f"refinement_ablation_summary_{scale}_{timestamp}.json"
    _save_json(refinement_path, refinement_summary)
    manifest["outputs"]["refinement_ablation"] = str(refinement_path)
    _print_table(refinement_summary)

    # 3. Personalization
    if not skip_personalization:
        print("\n=== Step 3: Personalization evaluation ===")
        try:
            pers_report = run_personalization_evaluation(
                dataset_name=dataset,
                modes=["bm25", "embedding", "hybrid_parallel"],
                top_k=top_k,
                max_queries=max_queries,
                warmup_queries=5,
            )
            pers_report["scale"] = scale
            pers_summary = {
                "scale": scale,
                "timestamp": pers_report["timestamp"],
                "dataset_name": pers_report["dataset_name"],
                "eval_protocol": pers_report["eval_protocol"],
                "deltas_personalized_vs_baseline": pers_report["deltas_personalized_vs_baseline"],
                "baseline": {"modes": pers_report["baseline"]["modes"]},
                "personalized": {"modes": pers_report["personalized"]["modes"]},
            }
            pers_path = output_dir / f"personalization_ablation_summary_{scale}_{timestamp}.json"
            _save_json(pers_path, pers_summary)
            manifest["outputs"]["personalization"] = str(pers_path)
        except Exception as exc:
            print(f"Personalization eval skipped: {exc}")
            manifest["outputs"]["personalization"] = f"skipped: {exc}"

    # 4. Vector store
    print("\n=== Step 4: Vector store evaluation ===")
    vs_report = run_vector_store_evaluation(
        dataset_name=dataset,
        top_k=top_k,
        max_queries=max_queries,
    )
    vs_report["scale"] = scale
    vs_path = output_dir / f"vector_store_ablation_{scale}_{timestamp}.json"
    _save_json(vs_path, vs_report)
    manifest["outputs"]["vector_store"] = str(vs_path)

    # 5. RAG
    if not skip_rag and os.environ.get("GEMINI_API_KEY"):
        print("\n=== Step 5: RAG evaluation ===")
        try:
            rag_report = run_rag_evaluation(
                dataset_name=dataset,
                max_queries=rag_max_queries,
                top_k=top_k,
            )
            rag_report["scale"] = scale
            rag_path = output_dir / f"rag_eval_{scale}_{timestamp}.json"
            _save_json(rag_path, rag_report)
            manifest["outputs"]["rag"] = str(rag_path)
        except Exception as exc:
            print(f"RAG eval skipped: {exc}")
            manifest["outputs"]["rag"] = f"skipped: {exc}"
    else:
        print("\n=== Step 5: RAG evaluation skipped (no GEMINI_API_KEY or --skip-rag) ===")
        manifest["outputs"]["rag"] = "skipped"

    # 6. Clustering
    if not skip_cluster:
        print("\n=== Step 6: Clustering evaluation ===")
        try:
            cluster_report = run_cluster_evaluation()
            cluster_report["scale"] = scale
            cluster_path = output_dir / f"cluster_eval_{scale}_{timestamp}.json"
            _save_json(cluster_path, cluster_report)
            manifest["outputs"]["clustering"] = str(cluster_path)
        except Exception as exc:
            print(f"Cluster eval skipped: {exc}")
            manifest["outputs"]["clustering"] = f"skipped: {exc}"

    # 7. Visualizations
    print("\n=== Step 7: Generating charts and summary ===")
    try:
        viz_result = generate_all(output_dir)
    except Exception as exc:
        print(f"Visualization failed: {exc}")
        viz_result = {"error": str(exc)}
    manifest["visualization"] = viz_result

    manifest_path = output_dir / f"full_eval_manifest_{scale}_{timestamp}.json"
    _save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full IR evaluation pipeline")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="full", choices=["dev", "preval", "full"])
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--modes", default=",".join(VALID_REPRESENTATION_MODES))
    parser.add_argument("--output-dir", default=str(ROOT / "evaluation_results"))
    parser.add_argument("--skip-rag", action="store_true")
    parser.add_argument("--skip-personalization", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    parser.add_argument("--rag-max-queries", type=int, default=20)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    output_dir = Path(args.output_dir)

    print(
        "Full evaluation requires: preprocessing, retrieval, refinement, personalization, "
        "clustering, (optional) RAG services running with full index."
    )

    manifest = run_full_evaluation(
        dataset=args.dataset,
        scale=args.scale,
        max_queries=args.max_queries,
        top_k=args.top_k,
        modes=modes,
        output_dir=output_dir,
        skip_rag=args.skip_rag,
        skip_personalization=args.skip_personalization,
        skip_cluster=args.skip_cluster,
        rag_max_queries=args.rag_max_queries,
    )
    print(f"\nDone. Manifest: {output_dir / 'full_eval_manifest_*'}")
    print(json.dumps(manifest.get("visualization", {}), indent=2))


if __name__ == "__main__":
    main()
