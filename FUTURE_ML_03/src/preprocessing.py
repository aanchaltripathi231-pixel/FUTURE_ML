from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "will", "with", "can", "into", "using", "use",
    "plus", "should", "we", "our", "they", "them", "you", "your",
}


@lru_cache(maxsize=1)
def ensure_nltk_resources() -> bool:
    """Return True when NLTK tokenizers and stopwords are locally available."""
    resources = {
        "tokenizers/punkt": None,
        "tokenizers/punkt_tab": None,
        "corpora/stopwords": None,
    }
    for resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            return False
    return True


def lowercase_text(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> List[str]:
    if ensure_nltk_resources():
        return word_tokenize(text)
    return re.findall(r"[a-zA-Z]+", text)


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    if ensure_nltk_resources():
        stop_words = set(stopwords.words("english"))
    else:
        stop_words = FALLBACK_STOPWORDS
    return [token for token in tokens if token not in stop_words and token.isalpha()]


def preprocess_text(text: str) -> str:
    """Run the main text cleaning pipeline and return a cleaned string."""
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    tokens = tokenize_text(text)
    filtered_tokens = remove_stopwords(tokens)
    return " ".join(filtered_tokens)
