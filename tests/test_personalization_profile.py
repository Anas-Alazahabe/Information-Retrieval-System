"""Unit tests for personalization profile term weighting."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from personalization_service.app.core.profile_builder import (
    terms_from_click,
    terms_from_query,
)
from shared.ir_config import CLICK_EVENT_WEIGHT, QUERY_EVENT_WEIGHT


class TestProfileBuilder:
    def test_query_terms_use_query_weight(self):
        terms = terms_from_query("machine learning basics")
        assert terms["machine"] == QUERY_EVENT_WEIGHT
        assert terms["learning"] == QUERY_EVENT_WEIGHT

    def test_click_terms_use_higher_weight(self):
        terms = terms_from_click("diabetes insulin treatment guide")
        assert terms["diabetes"] == CLICK_EVENT_WEIGHT
        assert CLICK_EVENT_WEIGHT > QUERY_EVENT_WEIGHT

    def test_empty_text_returns_empty(self):
        assert terms_from_query("   ") == {}
        assert terms_from_click("") == {}
