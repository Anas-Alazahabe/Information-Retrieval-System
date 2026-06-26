import argparse
import io
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

# إجبار بايثون على استخدام utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.index_builder import IndexBuilder
from shared.index_checkpoint import (
    build_run_config,
    clear_checkpoint,
    configs_compatible,
    load_checkpoint,
    new_run_id,
    print_indexing_status,
    save_checkpoint,
)
from shared.ir_config import (
    DATASET_NAME,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PREPROCESS_FLAGS,
    get_max_docs_for_scale,
    preprocess_batch_url,
)


def _patch_ir_datasets_tsv_utf8() -> None:
    """Force UTF-8 when reading MS MARCO TSV on Windows (avoids cp1252 decode errors)."""
    import ir_datasets.formats.tsv as tsv_mod

    if getattr(tsv_mod.FileLineIter, "_utf8_patched", False):
        return

    def _open_utf8_stream(self, raw_stream):
        return io.TextIOWrapper(raw_stream, encoding="utf-8", errors="replace")

    def _patched_next(self):
        if self.stop is not None and self.start >= self.stop:
            self.ctxt.close()
            raise StopIteration
        if self.stream is None:
            if isinstance(self.dlc, list):
                self.stream = _open_utf8_stream(
                    self, self.ctxt.enter_context(self.dlc[self.stream_idx].stream())
                )
            else:
                self.stream = _open_utf8_stream(
                    self, self.ctxt.enter_context(self.dlc.stream())
                )
        line = ""
        while self.pos < self.start:
            line = self.stream.readline()
            if line != "\n":
                self.pos += 1
        if line == "":
            if isinstance(self.dlc, list):
                self.stream_idx += 1
                if self.stream_idx < len(self.dlc):
                    self.stream = _open_utf8_stream(
                        self,
                        self.ctxt.enter_context(self.dlc[self.stream_idx].stream()),
                    )
                    line = self.stream.readline()
                else:
                    raise StopIteration()
            else:
                raise StopIteration()
        self.start += self.step
        return line

    tsv_mod.FileLineIter.__next__ = _patched_next
    tsv_mod.FileLineIter._utf8_patched = True


class DatasetIndexer:
    """Index ir_datasets documents with optional checkpoint/resume."""

    def __init__(self, dataset_name: str = DATASET_NAME):
        import ir_datasets

        _patch_ir_datasets_tsv_utf8()
        self.dataset_name = dataset_name
        self.dataset = ir_datasets.load(dataset_name)
        self.builder = IndexBuilder()
        self._interrupt_requested = False
        self._checkpoint_meta: dict | None = None
        self._index_dir = INDEX_DIR
        self._embedding_model_path = EMBEDDING_MODEL

    def _request_stop(self, signum, frame):
        del signum, frame
        if self._interrupt_requested:
            print("\nForce quit — checkpoint already saved on first interrupt.")
            raise SystemExit(1)
        self._interrupt_requested = True
        print("\nStop requested (Ctrl+C). Finishing current batch, then saving checkpoint...")

    def _skip_documents(self, docs_iter, count: int) -> None:
        if count <= 0:
            return
        print(f"Skipping first {count:,} documents (resume)...")
        skipped = 0
        while skipped < count:
            try:
                next(docs_iter)
                skipped += 1
                if skipped % 10_000 == 0:
                    print(f"  skipped {skipped:,} / {count:,}")
            except StopIteration:
                raise RuntimeError(
                    f"Dataset ended while skipping; checkpoint expects {count:,} docs"
                ) from None
        print(f"Resume position: doc #{count + 1:,}")

    def _write_checkpoint(self, status: str = "in_progress") -> None:
        if not self._checkpoint_meta:
            return
        meta = save_checkpoint(self._index_dir, self._checkpoint_meta, self.builder, status=status)
        self._checkpoint_meta = meta
        print(
            f"Checkpoint saved ({status}): {meta['docs_processed']:,} docs "
            f"-> {self._index_dir}/.checkpoint/"
        )

    def process_and_index(
        self,
        batch_size: int = 500,
        max_docs: int | None = None,
        index_scale_mode: str = "dev",
        index_dir: str = INDEX_DIR,
        *,
        resume: bool = True,
        fresh: bool = False,
        checkpoint_every: int = 1,
    ):
        self._index_dir = index_dir
        if max_docs is None:
            max_docs = get_max_docs_for_scale(index_scale_mode)
        if max_docs is None:
            max_docs = float("inf")

        run_config = build_run_config(
            dataset_name=self.dataset_name,
            max_docs=None if max_docs == float("inf") else int(max_docs),
            embedding_model=self._embedding_model_path,
            index_dir=index_dir,
            index_scale_mode=index_scale_mode,
            batch_size=batch_size,
        )

        if fresh:
            clear_checkpoint(index_dir)
            print("Fresh run: cleared any previous checkpoint.")

        loaded_meta, loaded_builder = load_checkpoint(index_dir)
        docs_already_indexed = 0

        if loaded_meta and loaded_builder and resume and not fresh:
            ok, reason = configs_compatible(
                loaded_meta.get("run_config", {}),
                run_config,
                docs_processed=loaded_meta.get("docs_processed", 0),
            )
            if not ok:
                raise RuntimeError(
                    f"Cannot resume checkpoint: {reason}. "
                    "Use --fresh to start a new run or match the previous settings."
                )
            target = loaded_meta.get("run_config", {}).get("max_docs")
            current_target = run_config.get("max_docs")
            if (
                loaded_meta.get("status") == "completed"
                and target is not None
                and current_target is not None
                and current_target > target
            ):
                print(
                    f"Extending completed index ({loaded_meta['docs_processed']:,} docs) "
                    f"toward new target {current_target:,}."
                )
            elif loaded_meta.get("status") == "completed" and current_target == target:
                print(
                    f"Checkpoint already complete ({loaded_meta['docs_processed']:,} docs). "
                    "Rebuilding final artifacts..."
                )
                manifest = loaded_builder.save(
                    index_dir=index_dir,
                    dataset_name=self.dataset_name,
                    embedding_model=self._embedding_model_path,
                    index_scale_mode=index_scale_mode,
                    max_docs_cap=run_config.get("max_docs"),
                )
                return manifest

            self.builder = loaded_builder
            docs_already_indexed = self.builder.total_docs
            self._checkpoint_meta = {**loaded_meta, "run_config": run_config}
            print(
                f"Resuming run {loaded_meta.get('run_id', '?')}: "
                f"{docs_already_indexed:,} docs already indexed, "
                f"target {run_config.get('max_docs', 'unlimited')}."
            )
        else:
            if loaded_meta and not fresh and not resume:
                print("Ignoring existing checkpoint (--no-resume). Starting from document 1.")
            self.builder = IndexBuilder()
            self._checkpoint_meta = {
                "run_id": new_run_id(),
                "run_config": run_config,
                "started_at": None,
            }

        if docs_already_indexed >= max_docs:
            print(f"Target already reached ({docs_already_indexed:,} >= {max_docs:,}). Saving index...")
            manifest = self.builder.save(
                index_dir=index_dir,
                dataset_name=self.dataset_name,
                embedding_model=self._embedding_model_path,
                index_scale_mode=index_scale_mode,
                max_docs_cap=run_config.get("max_docs"),
            )
            self._write_checkpoint(status="completed")
            clear_checkpoint(index_dir)
            return manifest

        signal.signal(signal.SIGINT, self._request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._request_stop)

        import requests
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model from: {self._embedding_model_path}")
        embedding_model = SentenceTransformer(self._embedding_model_path)
        print("Loaded embedding model successfully.")

        docs = self.dataset.docs_iter()
        self._skip_documents(docs, docs_already_indexed)

        loaded_docs = docs_already_indexed
        batches_since_checkpoint = 0

        if not self._checkpoint_meta.get("started_at"):
            from datetime import datetime, timezone

            self._checkpoint_meta["started_at"] = datetime.now(timezone.utc).isoformat()

        print(
            f"Indexing {self.dataset_name}: "
            f"{loaded_docs:,} done, target {max_docs:,}, batch {batch_size}"
        )

        try:
            while loaded_docs < max_docs and not self._interrupt_requested:
                batch_raw_texts = []
                batch_ids = []

                while len(batch_raw_texts) < batch_size and loaded_docs < max_docs:
                    try:
                        doc = next(docs)
                        doc_text = getattr(doc, "text", getattr(doc, "body", ""))
                        batch_raw_texts.append(str(doc_text).strip())
                        batch_ids.append(doc.doc_id)
                        loaded_docs += 1
                    except StopIteration:
                        break

                if not batch_raw_texts:
                    break

                try:
                    payload = {"texts": batch_raw_texts, **PREPROCESS_FLAGS}
                    response = requests.post(preprocess_batch_url(), json=payload, timeout=120)
                    response.raise_for_status()
                    cleaned_texts = response.json()["results"]
                except Exception as exc:
                    print(f"Preprocessing failed, falling back to raw: {exc}")
                    cleaned_texts = batch_raw_texts

                cleaned_for_embedding = [
                    " ".join(doc) if isinstance(doc, list) else str(doc)
                    for doc in cleaned_texts
                ]
                vectors = embedding_model.encode(
                    cleaned_for_embedding, normalize_embeddings=True
                ).tolist()

                count = self.builder.add_documents(batch_ids, cleaned_texts, vectors)
                print(f"Indexed batch of {count} documents ({self.builder.total_docs:,} total).")

                batches_since_checkpoint += 1
                if batches_since_checkpoint >= checkpoint_every:
                    self._write_checkpoint(status="in_progress")
                    batches_since_checkpoint = 0

            if self._interrupt_requested:
                self._write_checkpoint(status="paused")
                print("Paused. Run the same command again to resume.")
                return None

            manifest = self.builder.save(
                index_dir=index_dir,
                dataset_name=self.dataset_name,
                embedding_model=self._embedding_model_path,
                index_scale_mode=index_scale_mode,
                max_docs_cap=run_config.get("max_docs"),
            )
            self._write_checkpoint(status="completed")
            clear_checkpoint(index_dir)
            print(f"Manifest written: {manifest.get('timestamp')}")
            return manifest

        except Exception:
            print("Error during indexing — saving checkpoint before exit.")
            self._write_checkpoint(status="paused")
            raise


def main():
    parser = argparse.ArgumentParser(description="Build IR index from ir_datasets collection")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--scale", default="dev", choices=["dev", "preval", "full"])
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-docs", type=int, default=None, help="Override scale cap")
    parser.add_argument("--index-dir", default=INDEX_DIR)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear checkpoint and start a new run from document 1",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint even if present (starts from document 1)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="Save checkpoint every N batches (default: 1)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print checkpoint/index progress and exit (no indexing)",
    )
    args = parser.parse_args()

    max_docs = args.max_docs
    if max_docs is None:
        max_docs = get_max_docs_for_scale(args.scale)

    if args.status:
        current_config = build_run_config(
            dataset_name=args.dataset,
            max_docs=max_docs,
            embedding_model=EMBEDDING_MODEL,
            index_dir=args.index_dir,
            index_scale_mode=args.scale,
            batch_size=args.batch_size,
        )
        print_indexing_status(args.index_dir, current_run_config=current_config)
        return

    indexer = DatasetIndexer(args.dataset)
    indexer.process_and_index(
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        index_scale_mode=args.scale,
        index_dir=args.index_dir,
        resume=not args.no_resume,
        fresh=args.fresh,
        checkpoint_every=max(1, args.checkpoint_every),
    )


if __name__ == "__main__":
    main()
