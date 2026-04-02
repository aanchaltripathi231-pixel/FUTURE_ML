from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_support_tickets.csv"

CATEGORY_MAPPING = {
    "Billing inquiry": "Billing",
    "Refund request": "Billing",
    "Technical issue": "Technical Issue",
    "Cancellation request": "Account",
    "Product inquiry": "General Query",
}

PRIORITY_MAPPING = {
    "Critical": "High",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}

URGENT_KEYWORDS = {
    "urgent",
    "immediately",
    "asap",
    "critical",
    "not working",
    "unable",
    "cant",
    "cannot",
    "error",
    "failed",
    "broken",
    "refund",
}

MEDIUM_KEYWORDS = {
    "issue",
    "problem",
    "delay",
    "slow",
    "help",
    "support",
    "question",
}


def _ensure_nltk_resource(resource_path: str, resource_name: str) -> bool:
    try:
        find(resource_path)
        return True
    except LookupError:
        return False


STOPWORDS_READY = _ensure_nltk_resource("corpora/stopwords", "stopwords")
WORDNET_READY = _ensure_nltk_resource("corpora/wordnet", "wordnet")
OMW_READY = _ensure_nltk_resource("corpora/omw-1.4", "omw-1.4")

TOKENIZER = TreebankWordTokenizer()
LEMMATIZER = WordNetLemmatizer() if WORDNET_READY else None


def get_stop_words() -> set[str]:
    if STOPWORDS_READY:
        try:
            return set(stopwords.words("english"))
        except LookupError:
            pass
    return set(ENGLISH_STOP_WORDS)


STOP_WORDS = get_stop_words()


def normalize_category(ticket_type: str) -> str:
    return CATEGORY_MAPPING.get(str(ticket_type).strip(), "General Query")


def derive_priority_from_text(ticket_text: str) -> str:
    text = ticket_text.lower()
    if any(keyword in text for keyword in URGENT_KEYWORDS):
        return "High"
    if any(keyword in text for keyword in MEDIUM_KEYWORDS):
        return "Medium"
    return "Low"


def normalize_priority(priority_value: str, ticket_text: str) -> str:
    mapped = PRIORITY_MAPPING.get(str(priority_value).strip())
    if mapped:
        return mapped
    return derive_priority_from_text(ticket_text)


def build_ticket_text(subject: str, description: str) -> str:
    return f"{subject} {description}".strip()


def preprocess_text(text: str, lemmatize: bool = True) -> str:
    text = str(text).lower()
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = TOKENIZER.tokenize(text)
    cleaned_tokens: list[str] = []

    for token in tokens:
        if token in STOP_WORDS or len(token) <= 2:
            continue
        if lemmatize and LEMMATIZER is not None and OMW_READY:
            token = LEMMATIZER.lemmatize(token)
        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def load_raw_data(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DATA_PATH
    return pd.read_csv(path)


def load_and_prepare_data(data_path: Path | None = None) -> pd.DataFrame:
    df = load_raw_data(data_path).copy()

    df["ticket_text"] = (
        df[["Ticket Subject", "Ticket Description"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )
    df["clean_ticket_text"] = df["ticket_text"].apply(preprocess_text)
    df["category_label"] = df["Ticket Type"].apply(normalize_category)
    df["priority_label"] = df.apply(
        lambda row: normalize_priority(row.get("Ticket Priority", ""), row["ticket_text"]),
        axis=1,
    )

    df = df[df["clean_ticket_text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["ticket_text", "category_label", "priority_label"])
    df = df.reset_index(drop=True)
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = df["category_label"].astype(str) + "__" + df["priority_label"].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    prepared_df = load_and_prepare_data()
    print(prepared_df[["ticket_text", "clean_ticket_text", "category_label", "priority_label"]].head())
    print("\nCategory distribution:")
    print(prepared_df["category_label"].value_counts())
    print("\nPriority distribution:")
    print(prepared_df["priority_label"].value_counts())
