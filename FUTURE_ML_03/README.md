# Resume / Candidate Screening System

This project is a simple machine learning system that reads resumes, compares them with a given job description, and ranks candidates based on how well they match the role.

## Problem Statement

Hiring teams often spend a lot of time manually reading resumes. The goal here is to make that first screening step faster and more consistent by using NLP techniques to check relevance, match skills, and highlight gaps.

## What This Project Does

The system takes a set of resumes and one job description, cleans the text, extracts useful skills, creates TF-IDF vectors, calculates cosine similarity, and combines that with skill matching to score each candidate.

## Main Features

- Cleans and prepares resume and job description text
- Extracts important skills using spaCy and a basic keyword list
- Converts text into numerical form using TF-IDF
- Compares resumes with the job description using similarity scores
- Matches skills and gives weight to important ones
- Ranks candidates based on how well they fit the role
- Shows which skills are matched and which are missing
- Displays top 5 best-matching candidates
- Includes a notebook for a clear step-by-step view

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

- Sample resumes and a job description are included so the project runs smoothly from start to finish.
- You can replace the files in the `data/` folder with your own dataset anytime.
- The code is kept simple and modular so it’s easy to understand and modify.
