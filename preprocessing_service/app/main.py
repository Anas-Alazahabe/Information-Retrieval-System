import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.cleaner import TextCleaner
from shared.ir_config import PREPROCESS_FLAGS

app = FastAPI(
    title="Preprocessing Service",
    version="2.0",
)

cleaner = TextCleaner()


class PreprocessRequest(BaseModel):
    """طلب معالجة نص واحد."""

    text: str
    use_stemming: bool = PREPROCESS_FLAGS["use_stemming"]
    use_lemmatization: bool = PREPROCESS_FLAGS["use_lemmatization"]
    remove_stopwords: bool = PREPROCESS_FLAGS["remove_stopwords"]
    preserve_wh_words: bool = False


class BatchPreprocessRequest(BaseModel):
    """طلب معالجة دفعة نصوص."""

    texts: List[str]
    use_stemming: bool = PREPROCESS_FLAGS["use_stemming"]
    use_lemmatization: bool = PREPROCESS_FLAGS["use_lemmatization"]
    remove_stopwords: bool = PREPROCESS_FLAGS["remove_stopwords"]


@app.post("/preprocess")
def preprocess_text(request: PreprocessRequest):
    """معالجة نص مفرد وإرجاع التوكنز الناتجة."""
    processed_tokens = cleaner.process(
        text=request.text,
        use_stemming=request.use_stemming,
        use_lemmatization=request.use_lemmatization,
        remove_stop=request.remove_stopwords,
        preserve_wh_words=request.preserve_wh_words,
    )

    return {
        "tokens": processed_tokens,
        "count": len(processed_tokens),
    }


@app.post("/preprocess-batch")
def preprocess_batch(request: BatchPreprocessRequest):
    """معالجة دفعة نصوص دفعةً واحدة لتحسين الأداء."""
    batch_results = []

    for text in request.texts:
        tokens = cleaner.process(
            text=text,
            use_stemming=request.use_stemming,
            use_lemmatization=request.use_lemmatization,
            remove_stop=request.remove_stopwords,
        )
        batch_results.append(tokens)

    return {
        "results": batch_results,
        "batch_size": len(batch_results),
    }


@app.get("/health")
def health_check():
    """فحص صحة الخدمة وإظهار نمط المعالجة المفعّل."""
    return {
        "status": "healthy",
        "service": "preprocessing_service",
        "spacy_available": cleaner.spacy_available,
        "lemmatization_mode": cleaner.lemmatization_mode,
        "preprocess_defaults": PREPROCESS_FLAGS,
        "supports_preserve_wh_words": True,
    }
