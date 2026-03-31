from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from forecasting_pipeline import (
    clean_sales_data,
    engineer_features,
    evaluate_models,
    load_sales_data,
    recursive_forecast,
    time_based_split,
    train_models,
)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "raw" / "store_sales.csv"
    assert data_path.exists(), f"Dataset not found: {data_path}"

    raw_df = load_sales_data(data_path)
    assert not raw_df.empty, "Raw dataset is empty."
    assert {"Date", "Store", "Sales"}.issubset(raw_df.columns), "Required columns are missing."
    assert pd.api.types.is_datetime64_any_dtype(raw_df["Date"]), "Date column is not datetime."

    cleaned_df, cleaning_summary = clean_sales_data(raw_df)
    assert len(cleaned_df) <= len(raw_df), "Cleaning should not increase row count."
    assert cleaning_summary["rows_after_cleaning"] > 0, "Cleaned dataset has no rows."

    featured_df = engineer_features(cleaned_df)
    assert not featured_df.empty, "Feature engineering produced an empty dataset."
    required_features = {"lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"}
    assert required_features.issubset(featured_df.columns), "Expected engineered features are missing."

    train_df, test_df = time_based_split(featured_df, test_days=60)
    assert train_df["Date"].max() < test_df["Date"].min(), "Train/test split is not strictly time-based."

    trained_models = train_models(train_df)
    metrics_df, predictions = evaluate_models(trained_models, test_df)
    assert len(metrics_df) == 2, "Expected exactly two models in comparison."
    assert {"Model", "MAE", "RMSE"}.issubset(metrics_df.columns), "Metrics output is incomplete."
    assert all(value > 0 for value in metrics_df["RMSE"]), "RMSE values must be positive."

    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    assert best_model_name in predictions, "Missing predictions for best model."
    forecast_df = recursive_forecast(best_model, cleaned_df, horizon=30)
    assert len(forecast_df) == 30 * cleaned_df["Store"].nunique(), "Forecast output row count is incorrect."
    assert forecast_df["Date"].min() > cleaned_df["Date"].max(), "Forecast dates must be in the future."
    assert forecast_df["Sales"].notna().all(), "Forecast contains missing sales values."

    print("All project checks passed.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
