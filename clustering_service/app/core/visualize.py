import io
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_cluster_png(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
) -> bytes:
    """Render a scatter plot PNG from cached t-SNE coordinates."""
    unique_labels = np.unique(labels)
    centroids = np.array(
        [coords[labels == cluster_id].mean(axis=0) for cluster_id in unique_labels]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=40, alpha=0.7, cmap="viridis")
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        s=200,
        marker="o",
        edgecolors="white",
        linewidth=2,
    )
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
