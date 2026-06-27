"""CLI entry point for offline cluster precompute."""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clustering_service.app.core.precompute import run_precompute
from shared.ir_config import (
    CLUSTER_NUM_CLUSTERS_MAX,
    CLUSTER_VIZ_MAX_POINTS,
    INDEX_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute K-Means clusters and t-SNE visualization cache"
    )
    parser.add_argument("--index-dir", default=INDEX_DIR, help="Index directory path")
    parser.add_argument(
        "--max-k",
        type=int,
        default=CLUSTER_NUM_CLUSTERS_MAX,
        help="Maximum number of clusters",
    )
    parser.add_argument(
        "--viz-max-points",
        type=int,
        default=CLUSTER_VIZ_MAX_POINTS,
        help="Max points for t-SNE visualization subsample",
    )
    args = parser.parse_args()

    manifest = run_precompute(
        index_dir=args.index_dir,
        max_k=args.max_k,
        viz_max_points=args.viz_max_points,
    )
    print(
        f"Done: {manifest['total_docs']} docs, "
        f"{manifest['n_clusters']} clusters, "
        f"viz sample {manifest['viz_sample_size']}"
    )


if __name__ == "__main__":
    main()
