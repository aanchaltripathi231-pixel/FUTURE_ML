from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extraction import compare_skills, weighted_skill_score


def build_tfidf_vectors(job_description: str, resumes: list[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    corpus = [job_description] + resumes
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(corpus)
    return vectorizer, vectors


def compute_similarity_scores(vectors: np.ndarray) -> np.ndarray:
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity(resume_vectors, job_vector).flatten()


def calculate_final_score(
    similarity_score: float,
    matched_skills: list[str],
    total_jd_skills: int,
) -> dict:
    """Blend semantic similarity and weighted skill coverage into one score."""
    skill_match_ratio = len(matched_skills) / total_jd_skills if total_jd_skills else 0.0
    weighted_match_score = weighted_skill_score(matched_skills)
    final_score = (similarity_score * 0.6) + (skill_match_ratio * 0.25) + (weighted_match_score * 0.15 / 10)

    return {
        "similarity_score": round(float(similarity_score), 4),
        "skill_match_ratio": round(skill_match_ratio, 4),
        "weighted_skill_score": round(weighted_match_score, 4),
        "final_score": round(final_score, 4),
    }


def build_skill_gap_record(candidate_id: str, candidate_name: str, resume_skills: list[str], jd_skills: list[str]) -> dict:
    comparison = compare_skills(resume_skills, jd_skills)
    return {
        "candidate_id": candidate_id,
        "candidate_name": candidate_name,
        "matched_skills": ", ".join(comparison["matched_skills"]),
        "missing_skills": ", ".join(comparison["missing_skills"]),
        "matched_skill_count": len(comparison["matched_skills"]),
        "missing_skill_count": len(comparison["missing_skills"]),
    }
