from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays for models that need dense input."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


@dataclass(frozen=True)
class VectorizerConfig:
    name: str
    vectorizer: object


def get_vectorizers() -> dict[str, object]:
    common_kwargs = {
        "max_features": 2000,
        "ngram_range": (1, 2),
        "min_df": 2,
    }
    return {
        "tfidf": TfidfVectorizer(**common_kwargs),
        "bag_of_words": CountVectorizer(**common_kwargs),
    }


def build_pipeline(vectorizer, model_name: str, model) -> Pipeline:
    steps = [("vectorizer", vectorizer)]
    if model_name == "random_forest":
        steps.append(("to_dense", DenseTransformer()))
    steps.append(("model", model))
    return Pipeline(steps)
