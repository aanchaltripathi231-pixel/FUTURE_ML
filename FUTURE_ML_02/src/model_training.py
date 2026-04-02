from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.data_preprocessing import get_train_test_split, load_and_prepare_data
from src.evaluation import (
    calculate_metrics,
    plot_class_distribution,
    plot_confusion_matrix_chart,
    save_classification_report,
)
from src.feature_engineering import build_pipeline, get_vectorizers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
VISUALS_DIR = PROJECT_ROOT / "outputs" / "visuals"


def get_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=None,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    }


def train_task_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_name: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    vectorizers = get_vectorizers()
    models = get_models()
    comparison_rows: list[dict[str, object]] = []
    best_model = None
    best_predictions = None
    best_score = -1.0

    X_train = train_df["clean_ticket_text"]
    X_test = test_df["clean_ticket_text"]
    y_train = train_df[target_column]
    y_test = test_df[target_column]

    for vectorizer_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            pipeline = build_pipeline(clone(vectorizer), model_name, clone(model))
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = calculate_metrics(y_test, predictions)
            comparison_rows.append(
                {
                    "task": task_name,
                    "vectorizer": vectorizer_name,
                    "model": model_name,
                    **metrics,
                }
            )

            if metrics["f1_macro"] > best_score:
                best_score = metrics["f1_macro"]
                best_model = pipeline
                best_predictions = predictions

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["task", "f1_macro", "accuracy"],
        ascending=[True, False, False],
    )
    best_results_df = test_df[["ticket_text", target_column]].copy().reset_index(drop=True)
    best_results_df = best_results_df.rename(columns={target_column: f"actual_{task_name}"})
    best_results_df[f"predicted_{task_name}"] = best_predictions

    save_classification_report(
        y_test,
        best_predictions,
        METRICS_DIR / f"{task_name}_classification_report.csv",
    )
    plot_confusion_matrix_chart(
        y_test,
        best_predictions,
        labels=sorted(y_test.unique().tolist()),
        title=f"{task_name.title()} Confusion Matrix",
        output_path=VISUALS_DIR / f"{task_name}_confusion_matrix.png",
    )
    joblib.dump(best_model, MODELS_DIR / f"best_{task_name}_model.joblib")

    return comparison_df, best_results_df, best_model


def build_prediction_exports(
    df: pd.DataFrame,
    category_model,
    priority_model,
) -> pd.DataFrame:
    predictions_df = df[["ticket_text", "category_label", "priority_label"]].copy()
    predictions_df["predicted_category"] = category_model.predict(df["clean_ticket_text"])
    predictions_df["predicted_priority"] = priority_model.predict(df["clean_ticket_text"])
    predictions_df = predictions_df.rename(
        columns={
            "category_label": "actual_category",
            "priority_label": "actual_priority",
        }
    )
    predictions_df.to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)
    return predictions_df


def train_and_evaluate() -> dict[str, pd.DataFrame]:
    for directory in [MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR, VISUALS_DIR, PROJECT_ROOT / ".matplotlib"]:
        directory.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()
    train_df, test_df = get_train_test_split(df)

    plot_class_distribution(
        df["category_label"],
        "Support Ticket Category Distribution",
        VISUALS_DIR / "category_class_distribution.png",
        color="#2E7D32",
    )
    plot_class_distribution(
        df["priority_label"],
        "Support Ticket Priority Distribution",
        VISUALS_DIR / "priority_class_distribution.png",
        color="#C46210",
    )

    category_comparison_df, category_results_df, category_model = train_task_models(
        train_df=train_df,
        test_df=test_df,
        task_name="category",
        target_column="category_label",
    )
    priority_comparison_df, priority_results_df, priority_model = train_task_models(
        train_df=train_df,
        test_df=test_df,
        task_name="priority",
        target_column="priority_label",
    )

    model_comparison_df = pd.concat([category_comparison_df, priority_comparison_df], ignore_index=True)
    model_comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    category_comparison_df.to_csv(METRICS_DIR / "category_model_comparison.csv", index=False)
    priority_comparison_df.to_csv(METRICS_DIR / "priority_model_comparison.csv", index=False)

    holdout_predictions = pd.concat([category_results_df, priority_results_df.drop(columns=["ticket_text"])], axis=1)
    holdout_predictions.to_csv(PREDICTIONS_DIR / "holdout_predictions.csv", index=False)

    best_model_summary_df = (
        model_comparison_df.sort_values(by=["task", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("task", as_index=False)
        .first()
    )
    best_model_summary_df.to_csv(METRICS_DIR / "best_model_summary.csv", index=False)

    build_prediction_exports(df, category_model, priority_model)

    label_mapping_df = pd.DataFrame(
        [
            {"original_ticket_type": "Billing inquiry", "mapped_category": "Billing"},
            {"original_ticket_type": "Refund request", "mapped_category": "Billing"},
            {"original_ticket_type": "Technical issue", "mapped_category": "Technical Issue"},
            {"original_ticket_type": "Cancellation request", "mapped_category": "Account"},
            {"original_ticket_type": "Product inquiry", "mapped_category": "General Query"},
            {"original_priority": "Critical", "mapped_priority": "High"},
            {"original_priority": "High", "mapped_priority": "High"},
            {"original_priority": "Medium", "mapped_priority": "Medium"},
            {"original_priority": "Low", "mapped_priority": "Low"},
        ]
    )
    label_mapping_df.to_csv(METRICS_DIR / "label_mapping_reference.csv", index=False)

    return {
        "prepared_data": df,
        "train_data": train_df,
        "test_data": test_df,
        "model_comparison": model_comparison_df,
        "best_model_summary": best_model_summary_df,
    }


if __name__ == "__main__":
    results = train_and_evaluate()
    print("Training finished.\n")
    print(results["best_model_summary"].to_string(index=False))
