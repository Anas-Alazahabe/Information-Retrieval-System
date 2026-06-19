import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "query_refinement_service"))

from app.core.synonym_expander import expand_synonyms
from shared.ir_config import WH_WORDS


class TestExpandSynonyms:
    def test_content_word_has_synonyms(self):
        result = expand_synonyms(["car"], max_synonyms_per_term=3, max_total=5)
        assert result
        assert "car" not in result
        assert all(term.isalpha() and len(term) >= 2 for term in result)

    def test_wh_word_skipped(self):
        result = expand_synonyms(["what"], skip_terms=WH_WORDS, max_total=5)
        assert result == []

    def test_stopword_source_skipped(self):
        result = expand_synonyms(["the"], max_total=5)
        assert result == []

    def test_deterministic_output(self):
        tokens = ["car", "vehicle"]
        first = expand_synonyms(tokens, max_synonyms_per_term=2, max_total=4)
        second = expand_synonyms(tokens, max_synonyms_per_term=2, max_total=4)
        assert first == second

    def test_max_total_cap(self):
        result = expand_synonyms(
            ["car", "vehicle", "automobile"],
            max_synonyms_per_term=5,
            max_total=2,
        )
        assert len(result) <= 2

    def test_no_duplicate_with_query_tokens(self):
        result = expand_synonyms(["car"], max_synonyms_per_term=5, max_total=8)
        assert "car" not in result

    def test_empty_tokens(self):
        assert expand_synonyms([]) == []
