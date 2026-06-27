"""Tests for gzip index artifact I/O."""

import json
import os
import tempfile

from shared.index_json_io import artifact_exists, read_json_artifact, write_json_artifact


def test_write_and_read_gzip_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        payload = {"term": {"doc1": 1.0}}
        write_json_artifact(tmp, "metadata.json", payload, compress=True)
        assert not os.path.exists(os.path.join(tmp, "metadata.json"))
        assert os.path.exists(os.path.join(tmp, "metadata.json.gz"))
        assert artifact_exists(tmp, "metadata.json")
        loaded = read_json_artifact(tmp, "metadata.json")
        assert loaded == payload


def test_read_plain_json_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"a": 1}, handle)
        assert read_json_artifact(tmp, "metadata.json") == {"a": 1}
