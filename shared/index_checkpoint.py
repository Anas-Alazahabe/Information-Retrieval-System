"""Checkpoint save/load for resumable indexing runs."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.index_builder import IndexBuilder
from shared.ir_config import PREPROCESS_FLAGS

CHECKPOINT_VERSION = 1
CHECKPOINT_DIR_NAME = ".checkpoint"
META_FILENAME = "checkpoint.json"
STATE_FILENAME = "builder_state.pkl"
STATE_TMP_FILENAME = "builder_state.pkl.tmp"
STATE_BACKUP_FILENAME = "builder_state.pkl.bak"
LOCK_FILENAME = "indexer.lock"


def _pid_alive(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def acquire_indexer_lock(index_dir: str) -> Path:
    """Prevent two indexers from corrupting the same checkpoint."""
    lock = checkpoint_dir(index_dir) / LOCK_FILENAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_file():
        try:
            existing_pid = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and _pid_alive(existing_pid):
            raise RuntimeError(
                f"Another indexer is already running (PID {existing_pid}). "
                "Stop it before starting a new one."
            )
        lock.unlink(missing_ok=True)

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock), flags)
        try:
            os.write(fd, str(os.getpid()).encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError(
            "Another indexer is already running (lock file exists). "
            "Stop it before starting a new one."
        ) from exc
    return lock


def release_indexer_lock(lock: Optional[Path]) -> None:
    if lock is None or not lock.is_file():
        return
    try:
        if int(lock.read_text(encoding="utf-8").strip()) == os.getpid():
            lock.unlink()
    except (ValueError, OSError):
        pass


def checkpoint_dir(index_dir: str) -> Path:
    return Path(index_dir) / CHECKPOINT_DIR_NAME


def meta_path(index_dir: str) -> Path:
    return checkpoint_dir(index_dir) / META_FILENAME


def state_path(index_dir: str) -> Path:
    return checkpoint_dir(index_dir) / STATE_FILENAME


def build_run_config(
    *,
    dataset_name: str,
    max_docs: Optional[int],
    embedding_model: str,
    index_dir: str,
    index_scale_mode: str,
    batch_size: int,
    preprocess_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Fingerprint for a single indexing run (used to detect config changes)."""
    flags = preprocess_flags or PREPROCESS_FLAGS
    return {
        "dataset_name": dataset_name,
        "max_docs": max_docs,
        "embedding_model": embedding_model,
        "index_dir": str(index_dir),
        "index_scale_mode": index_scale_mode,
        "batch_size": batch_size,
        "preprocess_flags": dict(flags),
    }


def configs_compatible(
    stored: Dict[str, Any],
    current: Dict[str, Any],
    *,
    docs_processed: int = 0,
) -> Tuple[bool, str]:
    """Return whether an on-disk checkpoint can be resumed with the current settings."""
    for key in ("dataset_name", "embedding_model", "index_dir"):
        if stored.get(key) != current.get(key):
            return False, f"checkpoint {key} mismatch ({stored.get(key)!r} vs {current.get(key)!r})"

    stored_flags = stored.get("preprocess_flags") or {}
    current_flags = current.get("preprocess_flags") or {}
    if stored_flags != current_flags:
        return False, "checkpoint preprocessing flags differ from current settings"

    stored_max = stored.get("max_docs")
    current_max = current.get("max_docs")
    processed = int(docs_processed or stored.get("docs_processed", 0))

    if stored_max is not None and current_max is not None and current_max < processed:
        return False, (
            f"target max_docs ({current_max}) is below already indexed count ({processed})"
        )

    # Allow increasing the cap mid-run (e.g. 50K -> 200K).
    if stored.get("batch_size") != current.get("batch_size"):
        return False, "checkpoint batch_size differs; use --fresh to start over"

    return True, "ok"


def load_checkpoint(index_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[IndexBuilder]]:
    """Load checkpoint metadata and builder state if present."""
    meta_file = meta_path(index_dir)
    state_file = state_path(index_dir)
    if not meta_file.is_file():
        return None, None

    with meta_file.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    if meta.get("checkpoint_version") != CHECKPOINT_VERSION:
        return meta, None

    state_candidates = [state_file]
    backup = state_file.with_name(STATE_BACKUP_FILENAME)
    if backup.is_file():
        state_candidates.append(backup)

    last_error: Optional[Exception] = None
    for candidate in state_candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as handle:
                state = pickle.load(handle)
            builder = IndexBuilder()
            builder.load_state(state)
            return meta, builder
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(
            f"Checkpoint state file is unreadable ({last_error}). "
            "Wait for any running indexer to finish, or restore from backup."
        ) from last_error
    return meta, None


def save_checkpoint(
    index_dir: str,
    meta: Dict[str, Any],
    builder: IndexBuilder,
    *,
    status: str = "in_progress",
) -> Dict[str, Any]:
    """Persist builder state and run metadata."""
    ckpt = checkpoint_dir(index_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    meta = {
        **meta,
        "checkpoint_version": CHECKPOINT_VERSION,
        "status": status,
        "docs_processed": builder.total_docs,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    meta_out = meta_path(index_dir)
    meta_tmp = meta_out.with_suffix(".json.tmp")
    with meta_tmp.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    meta_tmp.replace(meta_out)

    state_out = state_path(index_dir)
    state_tmp = state_out.with_name(STATE_TMP_FILENAME)
    if state_out.is_file():
        backup = state_out.with_name(STATE_BACKUP_FILENAME)
        state_out.replace(backup)
    with state_tmp.open("wb") as handle:
        pickle.dump(builder.export_state(), handle)
    state_tmp.replace(state_out)

    return meta


def clear_checkpoint(index_dir: str) -> None:
    """Remove checkpoint files after a successful final save."""
    ckpt = checkpoint_dir(index_dir)
    for name in (META_FILENAME, STATE_FILENAME, STATE_TMP_FILENAME, STATE_BACKUP_FILENAME):
        path = ckpt / name
        if path.is_file():
            path.unlink()
    if ckpt.is_dir() and not any(ckpt.iterdir()):
        ckpt.rmdir()


def new_run_id() -> str:
    return str(uuid.uuid4())


def read_checkpoint_meta(index_dir: str) -> Optional[Dict[str, Any]]:
    """Read checkpoint metadata only (no builder state pickle)."""
    meta_file = meta_path(index_dir)
    if not meta_file.is_file():
        return None
    with meta_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_index_manifest(index_dir: str) -> Optional[Dict[str, Any]]:
    manifest_file = Path(index_dir) / "index_manifest.json"
    if not manifest_file.is_file():
        return None
    with manifest_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_indexing_status(
    index_dir: str,
    *,
    current_run_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a human-readable status report for checkpoint and/or saved index."""
    lines: list[str] = []
    index_path = Path(index_dir)
    lines.append(f"Index directory: {index_path.resolve()}")

    meta = read_checkpoint_meta(index_dir)
    manifest = _read_index_manifest(index_dir)

    if meta is None and manifest is None:
        lines.append("Status: no checkpoint and no saved index found.")
        lines.append("Next step: start indexing (preprocessing service must be running).")
        return "\n".join(lines)

    if meta is not None:
        run_config = meta.get("run_config") or {}
        docs_processed = int(meta.get("docs_processed", 0))
        target = run_config.get("max_docs")
        status = meta.get("status", "unknown")

        lines.append(f"Checkpoint: present ({checkpoint_dir(index_dir)})")
        lines.append(f"  run_id:        {meta.get('run_id', '?')}")
        lines.append(f"  status:        {status}")
        lines.append(f"  docs indexed:  {docs_processed:,}")
        if target is not None:
            lines.append(f"  target:        {target:,}")
            if target > 0:
                pct = min(100.0, 100.0 * docs_processed / target)
                remaining = max(0, target - docs_processed)
                lines.append(f"  progress:      {pct:.1f}% ({remaining:,} remaining)")
        else:
            lines.append("  target:        unlimited")
        if meta.get("started_at"):
            lines.append(f"  started_at:    {meta['started_at']}")
        if meta.get("updated_at"):
            lines.append(f"  updated_at:    {meta['updated_at']}")
        lines.append(f"  dataset:       {run_config.get('dataset_name', '?')}")
        lines.append(f"  scale mode:    {run_config.get('index_scale_mode', '?')}")
        lines.append(f"  batch_size:    {run_config.get('batch_size', '?')}")

        state_file = state_path(index_dir)
        if state_file.is_file():
            size_mb = state_file.stat().st_size / (1024 * 1024)
            lines.append(f"  state file:    {STATE_FILENAME} ({size_mb:.1f} MB)")
        elif meta.get("checkpoint_version") == CHECKPOINT_VERSION:
            lines.append("  state file:    MISSING (checkpoint metadata only)")

        if current_run_config:
            ok, reason = configs_compatible(
                run_config,
                current_run_config,
                docs_processed=docs_processed,
            )
            if ok:
                if status in ("paused", "in_progress"):
                    lines.append("Resume: compatible — re-run the same command (no --fresh).")
                elif status == "completed" and current_run_config.get("max_docs") != target:
                    new_target = current_run_config.get("max_docs")
                    if new_target and target and new_target > target:
                        lines.append(
                            f"Resume: can extend from {docs_processed:,} toward {new_target:,}."
                        )
            else:
                lines.append(f"Resume: incompatible with current CLI flags — {reason}")

    else:
        lines.append("Checkpoint: none")

    if manifest is not None:
        doc_count = manifest.get("document_count", "?")
        lines.append("Saved index: present (index_manifest.json)")
        lines.append(f"  documents:     {doc_count:,}" if isinstance(doc_count, int) else f"  documents:     {doc_count}")
        lines.append(f"  scale mode:    {manifest.get('index_scale_mode', '?')}")
        lines.append(f"  max_docs cap:  {manifest.get('max_docs_cap', '?')}")
        lines.append(f"  built_at:      {manifest.get('timestamp', '?')}")
        ann = manifest.get("ann_backend")
        if ann:
            lines.append(f"  ANN backend:   {ann}")

    return "\n".join(lines)


def print_indexing_status(
    index_dir: str,
    *,
    current_run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Print checkpoint/index progress without indexing."""
    print(format_indexing_status(index_dir, current_run_config=current_run_config))
