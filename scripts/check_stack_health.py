"""Quick health check for all IR stack services."""

import sys
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

SERVICES = [
    ("preprocessing", "http://127.0.0.1:8000/health"),
    ("retrieval", "http://127.0.0.1:8002/health"),
    ("refinement", "http://127.0.0.1:8003/health"),
    ("personalization", "http://127.0.0.1:8004/health"),
    ("clustering", "http://127.0.0.1:8005/health"),
    ("rag", "http://127.0.0.1:8006/health"),
]


def wait_retrieval(max_sec: int = 180) -> dict | None:
    for _ in range(max_sec // 5):
        try:
            resp = requests.get("http://127.0.0.1:8002/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("index_files_detected") and data.get("index_doc_count", 0) > 0:
                    return data
        except Exception:
            pass
        time.sleep(5)
    return None


def main() -> int:
    print("Waiting for retrieval to load indexes...")
    retrieval = wait_retrieval()
    if not retrieval:
        print("FAIL: retrieval not ready (index not loaded)")
        return 1

    for name, url in SERVICES:
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json()
            line = f"{name}: {resp.status_code} {data.get('status')}"
            if name == "retrieval":
                line += (
                    f" index={data.get('index_files_detected')}"
                    f" docs={data.get('index_doc_count')}"
                    f" ann={data.get('ann_backend')}"
                )
            elif name == "personalization":
                line += (
                    f" db={data.get('database_connected')}"
                    f" docs={data.get('documents_count')}"
                )
            elif name == "clustering":
                line += (
                    f" artifacts={data.get('cluster_artifacts_ready')}"
                    f" n={data.get('document_count')}"
                )
            elif name == "rag":
                line += (
                    f" gemini={data.get('gemini_configured')}"
                    f" db={data.get('database_connected')}"
                )
            print(line)
        except Exception as exc:
            print(f"{name}: ERR {exc}")
            return 1

    for mode in ("bm25", "embedding", "hybrid_parallel"):
        resp = requests.post(
            "http://127.0.0.1:8002/search",
            json={
                "query": "hospital information system",
                "representation_mode": mode,
                "top_k": 5,
            },
            timeout=120,
        )
        data = resp.json()
        n = len(data.get("results") or {})
        detail = data.get("detail")
        print(f"search_{mode}: {resp.status_code} results={n}" + (f" detail={detail}" if detail else ""))
        if resp.status_code != 200 or n == 0:
            return 1

    from shared.doc_store import fetch_document_texts

    resp = requests.post(
        "http://127.0.0.1:8002/search",
        json={"query": "health care", "representation_mode": "bm25", "top_k": 5},
        timeout=120,
    )
    ids = list((resp.json().get("results") or {}).keys())
    texts = fetch_document_texts(ids)
    found = sum(1 for doc_id in ids if texts.get(doc_id))
    print(f"mysql: {found}/{len(ids)} result docs found in DB")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
