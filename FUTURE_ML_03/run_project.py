from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_notebook import build_notebook
from src.preprocessing import preprocess_text
from src.ranking import get_top_candidates, rank_candidates
from src.scoring import build_skill_gap_record, build_tfidf_vectors, calculate_final_score, compute_similarity_scores
from src.skill_extraction import extract_skills
from src.utils import (
    DATA_DIR,
    NOTEBOOK_DIR,
    RANKING_DIR,
    VISUAL_DIR,
    create_top_candidates_chart,
    ensure_directories,
    load_job_description,
    load_resume_data,
    save_dataframe,
)


def main() -> None:
    ensure_directories()

    resumes_path = DATA_DIR / "resumes.csv"
    jd_path = DATA_DIR / "job_description.txt"

    resumes_df = load_resume_data(resumes_path)
    job_description = load_job_description(jd_path)
    cleaned_job_description = preprocess_text(job_description)
    jd_skills = extract_skills(job_description)

    resumes_df["cleaned_resume"] = resumes_df["resume_text"].apply(preprocess_text)
    resumes_df["resume_skills"] = resumes_df["resume_text"].apply(extract_skills)

    _, vectors = build_tfidf_vectors(cleaned_job_description, resumes_df["cleaned_resume"].tolist())
    similarity_scores = compute_similarity_scores(vectors)

    scored_rows = []
    skill_gap_rows = []

    for row, similarity_score in zip(resumes_df.to_dict("records"), similarity_scores):
        matched_skills = sorted(set(row["resume_skills"]).intersection(jd_skills))
        score_data = calculate_final_score(similarity_score, matched_skills, len(jd_skills))

        scored_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "similarity_score": score_data["similarity_score"],
                "skill_match_ratio": score_data["skill_match_ratio"],
                "weighted_skill_score": score_data["weighted_skill_score"],
                "final_score": score_data["final_score"],
                "matched_skills": ", ".join(matched_skills),
            }
        )

        skill_gap_rows.append(
            build_skill_gap_record(
                candidate_id=row["candidate_id"],
                candidate_name=row["candidate_name"],
                resume_skills=row["resume_skills"],
                jd_skills=jd_skills,
            )
        )

    ranked_candidates = rank_candidates(pd.DataFrame(scored_rows))
    skill_gap_df = pd.DataFrame(skill_gap_rows)
    top_candidates = get_top_candidates(ranked_candidates, top_n=5)

    ranked_output_path = RANKING_DIR / "ranked_candidates.csv"
    skill_gap_output_path = RANKING_DIR / "skill_gap_analysis.csv"
    chart_path = VISUAL_DIR / "top_candidates.png"
    notebook_path = NOTEBOOK_DIR / "resume_screening_analysis.ipynb"

    save_dataframe(ranked_candidates, ranked_output_path)
    save_dataframe(skill_gap_df, skill_gap_output_path)
    create_top_candidates_chart(ranked_candidates, chart_path, top_n=5)
    build_notebook(
        output_path=notebook_path,
        job_description=job_description,
        ranked_candidates=ranked_candidates,
        skill_gap_df=skill_gap_df,
        top_candidates=top_candidates,
    )

    print("Resume screening project completed successfully.")
    print(f"Rankings saved to: {ranked_output_path}")
    print(f"Skill gap analysis saved to: {skill_gap_output_path}")
    print(f"Chart saved to: {chart_path}")
    print(f"Notebook saved to: {notebook_path}")
    print("\nTop 5 candidates:")
    print(top_candidates[["rank", "candidate_name", "final_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
