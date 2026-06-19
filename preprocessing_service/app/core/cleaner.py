import logging
import string
import re
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.ir_config import WH_WORDS

logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class TextCleaner:
    """منظف نصوص موحد للوثائق والاستعلامات.

    يدعم:
    - تنظيف الروابط والضجيج
    - إزالة stopwords
    - stemming
    - lemmatization (مع fallback عند غياب spaCy)
    """

    def __init__(self):
        """تحميل أدوات التنظيف الأساسية ومحاولة تهيئة spaCy."""
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.nlp = None
        self.lemmatization_mode = "traditional_fallback"
        self._lemmatization_warning_logged = False

        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tok2vec", "attribute_ruler"])
            self.lemmatization_mode = "spacy"
            logger.info("spaCy model loaded for lemmatization.")
        except Exception as exc:
            self.nlp = None
            self.lemmatization_mode = "traditional_fallback"
            logger.warning("spaCy unavailable; lemmatization will use traditional fallback. (%s)", exc)

    @property
    def spacy_available(self) -> bool:
        """مؤشر يوضح توفر نموذج spaCy للاشتقاق الصرفي."""
        return self.nlp is not None

    def _keep_token(self, token: str, remove_stop: bool, preserve_wh_words: bool) -> bool:
        """يحدد ما إذا كان التوكن يُحفظ بعد إزالة stopwords."""
        if not remove_stop:
            return True
        if preserve_wh_words and token in WH_WORDS:
            return True
        return token not in self.stop_words

    def process(
        self,
        text: str,
        use_stemming: bool = False,
        use_lemmatization: bool = False,
        remove_stop: bool = True,
        preserve_wh_words: bool = False,
    ) -> list:
        """تنظيف النص وإرجاع قائمة توكنز بحسب الإعدادات المطلوبة."""

        if not text or not isinstance(text, str):
            return []

        if use_lemmatization and not self.nlp and not self._lemmatization_warning_logged:
            logger.warning(
                "use_lemmatization=True but spaCy is unavailable; using traditional tokenization."
            )
            self._lemmatization_warning_logged = True

        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        if use_lemmatization and self.nlp:
            doc = self.nlp(text)
            tokens = []

            for token in doc:
                if (
                    token.is_punct
                    or token.is_space
                    or token.is_digit
                    or token.like_num
                ):
                    continue

                lemma = token.lemma_.strip()
                lemma = lemma.translate(str.maketrans("", "", string.punctuation))

                if not lemma or len(lemma) < 2:
                    continue

                if not self._keep_token(lemma, remove_stop, preserve_wh_words):
                    continue

                tokens.append(lemma)

            if use_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]

            return tokens

        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        text = re.sub(r"\d+", " ", text)

        tokens = text.split()
        tokens = [token.strip() for token in tokens if token.strip()]

        if remove_stop:
            tokens = [
                token
                for token in tokens
                if self._keep_token(token, remove_stop, preserve_wh_words)
            ]

        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]

        tokens = [token for token in tokens if len(token) >= 2]

        return tokens
