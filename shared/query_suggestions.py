"""Prefix-based query suggestions over a sorted MS MARCO query catalog."""

import bisect
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_cache: Tuple[Optional[float], List[str]] = (None, [])


def load_catalog(path: str | Path) -> List[str]:
    """Load sorted query strings from disk with mtime-based caching."""
    global _cache
    catalog_path = Path(path)
    if not catalog_path.is_file():
        logger.warning("Query suggestions index not found: %s", catalog_path)
        return []

    mtime = catalog_path.stat().st_mtime
    if _cache[0] == mtime:
        return _cache[1]

    with catalog_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    queries = data.get("queries", [])
    _cache = (mtime, queries)
    return queries


def catalog_status(path: str | Path) -> Dict[str, object]:
    """Return whether the suggestion index is available and how many queries it holds."""
    catalog_path = Path(path)
    if not catalog_path.is_file():
        return {"loaded": False, "count": 0}
    return {"loaded": True, "count": len(load_catalog(catalog_path))}


def prefix_suggestions(catalog: List[str], prefix: str, limit: int = 5) -> List[str]:
    """Return up to `limit` catalog entries sharing a case-insensitive prefix."""
    if not catalog or not prefix:
        return []

    normalized = prefix.strip().lower()
    if not normalized:
        return []

    start = bisect.bisect_left(catalog, normalized)
    results: List[str] = []
    for index in range(start, len(catalog)):
        candidate = catalog[index]
        if not candidate.startswith(normalized):
            break
        results.append(candidate)
        if len(results) >= limit:
            break
    return results
