from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
RANKING_DIR = OUTPUT_DIR / "rankings"
VISUAL_DIR = OUTPUT_DIR / "visuals"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
CACHE_DIR = BASE_DIR / ".cache"
MPL_CONFIG_DIR = CACHE_DIR / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt


def ensure_directories() -> None:
    for directory in [DATA_DIR, OUTPUT_DIR, RANKING_DIR, VISUAL_DIR, NOTEBOOK_DIR, CACHE_DIR, MPL_CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_resume_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def load_job_description(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def save_dataframe(dataframe: pd.DataFrame, file_path: Path) -> None:
    dataframe.to_csv(file_path, index=False)


def save_json(data: dict, file_path: Path) -> None:
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_top_candidates_chart(ranked_candidates: pd.DataFrame, output_path: Path, top_n: int = 5) -> None:
    top_candidates = ranked_candidates.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(top_candidates["candidate_name"], top_candidates["final_score"], color="#2f6db3")
    plt.title("Top Resume Matches for the Job Description")
    plt.xlabel("Candidate")
    plt.ylabel("Final Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
