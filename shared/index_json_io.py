"""JSON index artifact I/O with optional gzip compression."""

import gzip
import json
import os
from typing import Any, Dict

COMPRESSIBLE_ARTIFACTS = (
    "vsm_index.json",
    "bm25_index.json",
    "embeddings_index.json",
    "metadata.json",
)


def artifact_paths(index_dir: str, filename: str) -> tuple[str, str]:
    plain = os.path.join(index_dir, filename)
    gz = plain + ".gz"
    return plain, gz


def read_json_artifact(index_dir: str, filename: str) -> Dict[str, Any]:
    plain, gz = artifact_paths(index_dir, filename)
    if os.path.exists(gz):
        with gzip.open(gz, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    if os.path.exists(plain):
        with open(plain, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def write_json_artifact(
    index_dir: str,
    filename: str,
    data: Any,
    *,
    compress: bool = True,
    indent: int | None = None,
) -> None:
    plain, gz = artifact_paths(index_dir, filename)
    if compress and filename in COMPRESSIBLE_ARTIFACTS:
        with gzip.open(gz, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=indent)
        if os.path.exists(plain):
            os.remove(plain)
        return
    with open(plain, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def artifact_exists(index_dir: str, filename: str) -> bool:
    plain, gz = artifact_paths(index_dir, filename)
    return os.path.exists(plain) or os.path.exists(gz)
