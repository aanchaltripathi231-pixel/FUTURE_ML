from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Set

import spacy


SKILL_KEYWORDS = {
    "python": 2.0,
    "sql": 1.8,
    "machine learning": 2.0,
    "ml": 2.0,
    "nlp": 1.8,
    "natural language processing": 1.8,
    "spacy": 1.4,
    "scikit-learn": 1.6,
    "sklearn": 1.6,
    "pandas": 1.3,
    "excel": 1.1,
    "communication": 1.2,
    "feature engineering": 1.4,
    "data preprocessing": 1.4,
    "tf-idf": 1.5,
    "cosine similarity": 1.5,
    "model evaluation": 1.4,
    "tensorflow": 1.2,
    "pytorch": 1.2,
    "aws": 1.1,
    "tableau": 1.0,
    "power bi": 1.0,
}


@lru_cache(maxsize=1)
def load_spacy_model():
    """Load spaCy model with a lightweight fallback if the small model is unavailable."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def _normalize_skill(skill: str) -> str:
    aliases = {
        "ml": "machine learning",
        "natural language processing": "nlp",
        "sklearn": "scikit-learn",
    }
    return aliases.get(skill, skill)


def extract_skills(text: str, skill_keywords: Dict[str, float] | None = None) -> List[str]:
    """Extract matched skills using both phrase lookup and token-level checks."""
    skill_keywords = skill_keywords or SKILL_KEYWORDS
    nlp = load_spacy_model()
    doc = nlp(text.lower())
    normalized_text = " ".join(token.text for token in doc)
    detected: Set[str] = set()

    for skill in skill_keywords:
        normalized_skill = _normalize_skill(skill)
        if " " in skill:
            if skill in normalized_text:
                detected.add(normalized_skill)
        else:
            if any(token.text == skill for token in doc):
                detected.add(normalized_skill)

    return sorted(detected)


def compare_skills(resume_skills: Iterable[str], jd_skills: Iterable[str]) -> Dict[str, List[str]]:
    resume_set = {_normalize_skill(skill) for skill in resume_skills}
    jd_set = {_normalize_skill(skill) for skill in jd_skills}
    matched = sorted(resume_set.intersection(jd_set))
    missing = sorted(jd_set.difference(resume_set))
    return {"matched_skills": matched, "missing_skills": missing}


def weighted_skill_score(matched_skills: Iterable[str], skill_keywords: Dict[str, float] | None = None) -> float:
    skill_keywords = skill_keywords or SKILL_KEYWORDS
    total = 0.0
    for skill in matched_skills:
        total += skill_keywords.get(skill, 1.0)
    return total
