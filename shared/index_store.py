"""طبقة وصول موحدة لقراءة ملفات الفهرسة من التخزين.

تسهّل هذه الطبقة فصل منطق الاسترجاع عن تفاصيل صيغة التخزين
بحيث يمكن استبدال JSON بمخزن آخر مستقبلًا.
"""

import json
import os
from typing import Any, Dict, Protocol, runtime_checkable

from shared.ir_config import ARTIFACT_FILES, INDEX_DIR


@runtime_checkable
class IndexStore(Protocol):
    """عقدة تجريدية (Protocol) لأي مخزن فهرسة."""

    def load_metadata(self) -> Dict[str, Any]: ...
    def load_vsm(self) -> Dict[str, Any]: ...
    def load_bm25(self) -> Dict[str, Any]: ...
    def load_embeddings(self) -> Dict[str, Any]: ...
    def load_manifest(self) -> Dict[str, Any]: ...
    def index_ready(self) -> bool: ...
    def get_index_mtime(self) -> float: ...


class JsonIndexStore:
    """تنفيذ `IndexStore` باستخدام ملفات JSON المحلية."""

    def __init__(self, index_dir: str = INDEX_DIR):
        """يهيئ المخزن على مسار فهارس محدد."""
        self.index_dir = index_dir

    def _read_json(self, filename: str) -> Dict[str, Any]:
        """يقرأ ملف JSON ويعيد محتواه أو قاموسًا فارغًا عند عدم وجوده."""
        path = os.path.join(self.index_dir, filename)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_metadata(self) -> Dict[str, Any]:
        """يحمّل بيانات الميتاداتا الخاصة بالفهرس."""
        return self._read_json("metadata.json")

    def load_vsm(self) -> Dict[str, Any]:
        """يحمّل فهرس VSM."""
        return self._read_json("vsm_index.json")

    def load_bm25(self) -> Dict[str, Any]:
        """يحمّل فهرس BM25."""
        return self._read_json("bm25_index.json")

    def load_embeddings(self) -> Dict[str, Any]:
        """يحمّل متجهات التضمين (Embeddings) للوثائق."""
        return self._read_json("embeddings_index.json")

    def load_manifest(self) -> Dict[str, Any]:
        """يحمّل ملف manifest الخاص بعملية البناء."""
        return self._read_json("index_manifest.json")

    def index_ready(self) -> bool:
        """يتحقق من جاهزية ملفات الفهرسة الأساسية."""
        return all(
            os.path.exists(os.path.join(self.index_dir, name))
            for name in ("metadata.json", "vsm_index.json", "bm25_index.json", "embeddings_index.json")
        )

    def get_index_mtime(self) -> float:
        """يعيد أحدث وقت تعديل بين ملفات الفهرسة."""
        mtimes = []
        for name in ARTIFACT_FILES:
            path = os.path.join(self.index_dir, name)
            if os.path.exists(path):
                mtimes.append(os.path.getmtime(path))
        return max(mtimes) if mtimes else 0.0
