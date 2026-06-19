"""Run baseline + refinement ablation evaluation and save summary report."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUTF8", "1")

from evaluation_service.app.main import run_evaluation
from shared.ir_config import EVAL_DATASET_NAME, VALID_REPRESENTATION_MODES

ABLATION_RUNS = [
    ("baseline", False, []),
    ("query_preprocess", True, ["query_preprocess"]),
    ("synonyms", True, ["synonyms"]),
    ("prf", True, ["prf"]),
    ("combined", True, ["query_preprocess", "prf", "synonyms"]),
]

METRIC_KEYS = ("map", "recall", "precision_at_10", "ndcg_at_10")


def _metric_delta(refined: float, baseline: float) -> float:
    return round(refined - baseline, 6)


def _build_summary(
    scale: str,
    reports: dict[str, dict],
) -> dict:
    baseline = reports["baseline"]
    summary = {
        "scale": scale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_name": baseline.get("dataset_name"),
        "max_queries": baseline.get("max_queries"),
        "top_k": baseline.get("top_k"),
        "runs": {},
        "deltas_vs_baseline": {},
    }

    for run_name, report in reports.items():
        summary["runs"][run_name] = {
            "use_refinement": report.get("use_refinement", False),
            "refinement_techniques": report.get("refinement_techniques", []),
            "modes": report.get("modes", {}),
        }

    baseline_modes = baseline.get("modes", {})
    for run_name, report in reports.items():
        if run_name == "baseline":
            continue
        summary["deltas_vs_baseline"][run_name] = {}
        for mode, metrics in report.get("modes", {}).items():
            base_metrics = baseline_modes.get(mode, {})
            summary["deltas_vs_baseline"][run_name][mode] = {
                metric: _metric_delta(metrics.get(metric, 0.0), base_metrics.get(metric, 0.0))
                for metric in METRIC_KEYS
            }

    return summary


def _print_table(summary: dict) -> None:
    print("\n=== Refinement ablation summary (delta vs baseline) ===")
    for run_name, mode_deltas in summary["deltas_vs_baseline"].items():
        print(f"\n[{run_name}]")
        for mode, deltas in mode_deltas.items():
            print(
                f"  {mode:16} MAP {deltas['map']:+.4f}  "
                f"Recall {deltas['recall']:+.4f}  "
                f"P@10 {deltas['precision_at_10']:+.4f}  "
                f"nDCG {deltas['ndcg_at_10']:+.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refinement ablation evaluation")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="preval", choices=["dev", "preval", "full"])
    parser.add_argument(
        "--modes",
        default=",".join(VALID_REPRESENTATION_MODES),
        help="Comma-separated representation modes",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(
        "Prerequisites: preprocessing (8000), retrieval (8002), and refinement (8003) "
        "must be running. Use a Stage B+ index (preval/full) for meaningful metrics."
    )

    reports: dict[str, dict] = {}
    for run_name, use_refinement, techniques in ABLATION_RUNS:
        print(f"\nRunning {run_name} ...")
        report = run_evaluation(
            dataset_name=args.dataset,
            representation_modes=modes,
            top_k=args.top_k,
            max_queries=args.max_queries,
            use_refinement=use_refinement,
            refinement_techniques=techniques,
        )
        report["scale"] = args.scale
        reports[run_name] = report

        if use_refinement:
            technique_slug = "-".join(techniques)
            filename = f"eval_refined_{technique_slug}_{args.scale}_{timestamp}.json"
        else:
            filename = f"eval_baseline_{args.scale}_{timestamp}.json"

        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved {output_path}")

    summary = _build_summary(args.scale, reports)
    summary_path = output_dir / f"refinement_ablation_summary_{args.scale}_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _print_table(summary)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
