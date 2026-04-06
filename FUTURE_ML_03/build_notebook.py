from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def _table_as_text(dataframe: pd.DataFrame) -> str:
    return dataframe.to_string(index=False)


def build_notebook(
    output_path: Path,
    job_description: str,
    ranked_candidates: pd.DataFrame,
    skill_gap_df: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> None:
    top_preview = _table_as_text(top_candidates[["rank", "candidate_name", "final_score"]])
    skill_gap_preview = _table_as_text(skill_gap_df[["candidate_name", "matched_skills", "missing_skills"]].head(5))

    notebook = {
        "cells": [
            _markdown_cell(
                """
                # Resume / Candidate Screening System

                This notebook walks through a simple NLP-based resume screening workflow.
                It compares resumes against a job description, extracts key skills, scores candidates,
                and produces a ranked list for HR or recruitment teams.
                """
            ),
            _markdown_cell(
                f"""
                ## Problem Statement

                Recruiters often review many resumes for one role. This project helps by:
                - cleaning and standardizing resume text
                - extracting important skills
                - measuring resume similarity with the job description
                - ranking candidates based on a combined score

                ## Sample Job Description

                {job_description}
                """
            ),
            _code_cell(
                """
                import pandas as pd

                resumes = pd.read_csv("data/resumes.csv")
                resumes.head()
                """
            ),
            _markdown_cell(
                """
                ## Preprocessing

                The preprocessing pipeline performs:
                - lowercasing
                - punctuation removal
                - tokenization
                - stopword removal
                """
            ),
            _code_cell(
                """
                from src.preprocessing import preprocess_text

                sample_text = resumes.loc[0, "resume_text"]
                print(preprocess_text(sample_text))
                """
            ),
            _markdown_cell(
                """
                ## Skill Extraction

                A keyword list and spaCy parsing are used to detect important job-related skills
                from both resumes and the job description.
                """
            ),
            _code_cell(
                """
                from src.skill_extraction import extract_skills

                resume_skills = resumes["resume_text"].apply(extract_skills)
                resume_skills.head()
                """
            ),
            _markdown_cell(
                """
                ## Scoring Logic

                The project combines two signals:
                - TF-IDF + cosine similarity
                - weighted skill match score
                """
            ),
            _markdown_cell(
                f"""
                ## Ranking Results

                Top 5 candidates from the latest run:

                ```text
                {top_preview}
                ```
                """
            ),
            _markdown_cell(
                f"""
                ## Skill Gap Snapshot

                Example skill gap output:

                ```text
                {skill_gap_preview}
                ```
                """
            ),
            _code_cell(
                """
                ranked_candidates = pd.read_csv("outputs/rankings/ranked_candidates.csv")
                skill_gap_analysis = pd.read_csv("outputs/rankings/skill_gap_analysis.csv")

                ranked_candidates.head()
                """
            ),
            _markdown_cell(
                """
                ## Final Note

                This workflow can be extended with larger datasets, better resume parsers,
                advanced embeddings, role-specific skill ontologies, and dashboard integration.
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
