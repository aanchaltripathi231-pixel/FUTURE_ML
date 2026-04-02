from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import build_ticket_text, preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_models():
    category_model = joblib.load(MODELS_DIR / "best_category_model.joblib")
    priority_model = joblib.load(MODELS_DIR / "best_priority_model.joblib")
    return category_model, priority_model


def predict_ticket(subject: str, description: str) -> dict[str, str]:
    ticket_text = build_ticket_text(subject, description)
    clean_text = preprocess_text(ticket_text)
    category_model, priority_model = load_models()

    category_prediction = category_model.predict([clean_text])[0]
    priority_prediction = priority_model.predict([clean_text])[0]

    return {
        "ticket_text": ticket_text,
        "predicted_category": category_prediction,
        "predicted_priority": priority_prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict category and priority for a new support ticket.")
    parser.add_argument("--subject", default="Login problem", help="Short ticket subject")
    parser.add_argument(
        "--text",
        default="I cannot access my account and need help immediately because the app keeps showing an error.",
        help="Ticket description",
    )
    parser.add_argument("--save", action="store_true", help="Save the prediction to outputs/predictions/new_ticket_prediction.csv")
    args = parser.parse_args()

    prediction = predict_ticket(args.subject, args.text)
    print(f"Ticket: {prediction['ticket_text']}")
    print(f"Predicted category: {prediction['predicted_category']}")
    print(f"Predicted priority: {prediction['predicted_priority']}")

    if args.save:
        output_path = PROJECT_ROOT / "outputs" / "predictions" / "new_ticket_prediction.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([prediction]).to_csv(output_path, index=False)
        print(f"Saved prediction to: {output_path}")


if __name__ == "__main__":
    main()
