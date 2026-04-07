# Resume / Candidate Screening System

This project is a small HR-tech style machine learning pipeline that reads resumes, compares them with a job description  and ranks candidates based on how well they fit the role.

## Problem Statement

Hiring teams often spend a lot of time manually reading resumes. The goal here is to make that first screening step faster and more consistent by using NLP techniques to check relevance, match skills, and highlight gaps.

## What This Project Does

The system takes a set of resumes and one job description, cleans the text, extracts useful skills, creates TF-IDF vectors, calculates cosine similarity, and combines that with skill matching to score each candidate.

## Main Features

- Resume and job description preprocessing
- Skill extraction using spaCy plus a predefined keyword list
- TF-IDF based text vectorization
- Cosine similarity scoring
- Weighted skill matching
- Candidate ranking
- Skill gap analysis showing matched and missing skills
- Top 5 candidate view
- Auto-generated Jupyter notebook summary

## How It Helps Businesses

For HR and recruitment teams, this kind of tool can reduce manual screening effort, help shortlist stronger applicants faster, and make the review process a bit more data-driven. It is useful as a first-pass screening layer before human review.

## Project Structure

```text
FUTURE_ML_03/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── skill_extraction.py
│   ├── scoring.py
│   ├── ranking.py
│   └── utils.py
├── outputs/
│   ├── rankings/
│   └── visuals/
├── run_project.py
├── build_notebook.py
├── requirements.txt
└── README.md
```

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

3. Run the full project:

```bash
python run_project.py
```

## Output Files

After running the project, you will get:

- `outputs/rankings/ranked_candidates.csv`
- `outputs/rankings/skill_gap_analysis.csv`
- `outputs/visuals/top_candidates.png`
- `notebooks/resume_screening_analysis.ipynb`

## Notes

- Sample resumes and a sample job description are already included so the project runs end-to-end.
- You can replace the sample files in `data/` with your own resume dataset later.
- The code is modular to make it easier to improve or expand easily.
