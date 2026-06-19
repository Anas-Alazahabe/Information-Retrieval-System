import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing_service.app.core.cleaner import TextCleaner
from shared.ir_config import PREPROCESS_FLAGS


@pytest.fixture
def cleaner():
    return TextCleaner()


def _process(cleaner, text):
    return cleaner.process(
        text=text,
        use_stemming=PREPROCESS_FLAGS["use_stemming"],
        use_lemmatization=PREPROCESS_FLAGS["use_lemmatization"],
        remove_stop=PREPROCESS_FLAGS["remove_stopwords"],
    )


def test_empty_text_returns_empty_list(cleaner):
    assert _process(cleaner, "") == []
    assert _process(cleaner, "   ") == []


def test_url_only_text_returns_empty_or_minimal(cleaner):
    tokens = _process(cleaner, "https://example.com/path")
    assert tokens == []


def test_punctuation_noise_returns_empty(cleaner):
    tokens = _process(cleaner, "!!! ???")
    assert tokens == []


def test_numbers_heavy_text_filtered(cleaner):
    tokens = _process(cleaner, "2024 100 3.14")
    assert tokens == []


def test_meaningful_text_produces_tokens(cleaner):
    tokens = _process(cleaner, "hospital information retrieval system")
    assert len(tokens) >= 2
    assert all(len(token) >= 2 for token in tokens)
