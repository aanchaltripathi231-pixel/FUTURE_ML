from __future__ import annotations

import pandas as pd


def rank_candidates(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = scored_candidates.sort_values(
        by=["final_score", "similarity_score", "weighted_skill_score"],
        ascending=False,
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def get_top_candidates(ranked_candidates: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    return ranked_candidates.head(top_n).copy()
