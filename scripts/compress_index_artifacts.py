"""Compress existing JSON index artifacts to .json.gz (streaming, low memory)."""

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.index_json_io import COMPRESSIBLE_ARTIFACTS
from shared.ir_config import INDEX_DIR


def _stream_gzip(plain: str, gz: str) -> None:
    with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)


def compress_index(index_dir: str) -> int:
    converted = 0
    for name in COMPRESSIBLE_ARTIFACTS:
        plain = os.path.join(index_dir, name)
        gz = plain + ".gz"
        if not os.path.exists(plain):
            continue
        if os.path.exists(gz):
            continue
        _stream_gzip(plain, gz)
        os.remove(plain)
        converted += 1
        print(f"Compressed {name} -> {name}.gz")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Gzip JSON index artifacts in place")
    parser.add_argument("--index-dir", default=INDEX_DIR)
    args = parser.parse_args()
    count = compress_index(args.index_dir)
    print(f"Done. {count} file(s) compressed under {args.index_dir}")


if __name__ == "__main__":
    main()
