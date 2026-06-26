"""Tests for indexing checkpoint compatibility and round-trip."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.index_builder import IndexBuilder
from shared.index_checkpoint import (
    build_run_config,
    clear_checkpoint,
    configs_compatible,
    format_indexing_status,
    load_checkpoint,
    meta_path,
    save_checkpoint,
)


@pytest.fixture
def tmp_index_dir(tmp_path):
    return str(tmp_path / "index_data")


def test_builder_state_round_trip():
    builder = IndexBuilder()
    builder.add_documents(
        ["d1", "d2"],
        [["alpha", "beta"], ["gamma"]],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    restored = IndexBuilder()
    restored.load_state(builder.export_state())
    assert restored.total_docs == 2
    assert "alpha" in restored.raw_inverted_index
    assert restored.doc_embeddings["d1"] == [0.1, 0.2]


def test_checkpoint_save_load(tmp_index_dir):
    builder = IndexBuilder()
    builder.add_documents(["d1"], [["hello"]], [[1.0, 0.0]])
    config = build_run_config(
        dataset_name="msmarco-passage",
        max_docs=50_000,
        embedding_model="/models/test",
        index_dir=tmp_index_dir,
        index_scale_mode="preval",
        batch_size=500,
    )
    meta = {"run_id": "test-run", "run_config": config}
    save_checkpoint(tmp_index_dir, meta, builder, status="in_progress")

    loaded_meta, loaded_builder = load_checkpoint(tmp_index_dir)
    assert loaded_meta is not None
    assert loaded_meta["docs_processed"] == 1
    assert loaded_meta["status"] == "in_progress"
    assert loaded_builder is not None
    assert loaded_builder.total_docs == 1

    clear_checkpoint(tmp_index_dir)
    assert load_checkpoint(tmp_index_dir) == (None, None)


def test_configs_compatible_rejects_dataset_change():
    stored = build_run_config(
        dataset_name="msmarco-passage",
        max_docs=50_000,
        embedding_model="m1",
        index_dir="/tmp/x",
        index_scale_mode="full",
        batch_size=500,
    )
    current = build_run_config(
        dataset_name="other-dataset",
        max_docs=50_000,
        embedding_model="m1",
        index_dir="/tmp/x",
        index_scale_mode="full",
        batch_size=500,
    )
    ok, reason = configs_compatible(stored, current)
    assert not ok
    assert "dataset_name" in reason


def test_format_status_no_artifacts(tmp_index_dir):
    report = format_indexing_status(tmp_index_dir)
    assert "no checkpoint" in report.lower()


def test_format_status_with_checkpoint_meta(tmp_index_dir):
    meta_path(tmp_index_dir).parent.mkdir(parents=True, exist_ok=True)
    config = build_run_config(
        dataset_name="msmarco-passage",
        max_docs=50_000,
        embedding_model="/models/test",
        index_dir=tmp_index_dir,
        index_scale_mode="preval",
        batch_size=500,
    )
    with meta_path(tmp_index_dir).open("w", encoding="utf-8") as handle:
        import json

        json.dump(
            {
                "run_id": "run-abc",
                "status": "paused",
                "docs_processed": 25_000,
                "run_config": config,
                "checkpoint_version": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T01:00:00+00:00",
            },
            handle,
        )

    report = format_indexing_status(tmp_index_dir, current_run_config=config)
    assert "25,000" in report
    assert "50.0%" in report
    assert "paused" in report
    assert "Resume: compatible" in report

def test_configs_allow_increasing_max_docs():
    stored = build_run_config(
        dataset_name="msmarco-passage",
        max_docs=50_000,
        embedding_model="m1",
        index_dir="/tmp/x",
        index_scale_mode="full",
        batch_size=500,
    )
    current = build_run_config(
        dataset_name="msmarco-passage",
        max_docs=200_000,
        embedding_model="m1",
        index_dir="/tmp/x",
        index_scale_mode="full",
        batch_size=500,
    )
    ok, _ = configs_compatible(stored, current, docs_processed=50_000)
    assert ok
