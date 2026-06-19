import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation_service.app.main import DEFAULT_REFINEMENT_TECHNIQUES, run_evaluation
from shared.ir_config import EVAL_DATASET_NAME, VALID_REPRESENTATION_MODES


def _build_output_name(use_refinement: bool, scale: str, techniques: list[str]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not use_refinement:
        return f"eval_baseline_{scale}_{timestamp}.json"
    technique_slug = "-".join(techniques) if techniques else "refined"
    return f"eval_refined_{technique_slug}_{scale}_{timestamp}.json"


def main():
    """تشغيل التقييم من سطر الأوامر وحفظ التقرير في `reports/`."""
    os.environ.setdefault("PYTHONUTF8", "1")
    parser = argparse.ArgumentParser(description="Run IR evaluation against retrieval service")
    parser.add_argument("--dataset", default=EVAL_DATASET_NAME)
    parser.add_argument("--scale", default="dev", choices=["dev", "preval", "full"])
    parser.add_argument(
        "--modes",
        default=",".join(VALID_REPRESENTATION_MODES),
        help="Comma-separated representation modes",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--output-dir", default=str(_ROOT / "evaluation_results"))
    parser.add_argument(
        "--use-refinement",
        action="store_true",
        help="Enable query refinement before each search",
    )
    parser.add_argument(
        "--refinement-techniques",
        default=",".join(DEFAULT_REFINEMENT_TECHNIQUES),
        help="Comma-separated refinement techniques when --use-refinement is set",
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    refinement_techniques = [
        t.strip() for t in args.refinement_techniques.split(",") if t.strip()
    ]

    report = run_evaluation(
        dataset_name=args.dataset,
        representation_modes=modes,
        top_k=args.top_k,
        max_queries=args.max_queries,
        use_refinement=args.use_refinement,
        refinement_techniques=refinement_techniques,
    )
    report["scale"] = args.scale

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = _build_output_name(args.use_refinement, args.scale, refinement_techniques)
    output_path = output_dir / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
