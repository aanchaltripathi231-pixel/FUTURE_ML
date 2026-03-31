from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "Store",
    "Promo",
    "Holiday",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "IsWeekend",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
]


@dataclass
class ForecastArtifacts:
    cleaned_data: pd.DataFrame
    featured_data: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    metrics: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    future_forecast: pd.DataFrame
    business_insights: List[str]


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (14, 6)


def load_sales_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw CSV data and standardize schema."""
    df = pd.read_csv(csv_path)
    df.columns = [column.strip().title() for column in df.columns]
    if "Date" not in df.columns or "Sales" not in df.columns:
        raise ValueError("Dataset must include at least Date and Sales columns.")

    if "Store" not in df.columns:
        df["Store"] = "All Stores"
    if "Promo" not in df.columns:
        df["Promo"] = 0
    if "Holiday" not in df.columns:
        df["Holiday"] = 0

    df["Date"] = pd.to_datetime(df["Date"])
    df["Store"] = df["Store"].astype(str)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def clean_sales_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Handle missing values, duplicates, and outliers in a business-friendly way."""
    cleaned = df.copy()
    initial_rows = len(cleaned)
    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned.groupby("Store")[column].transform(
                lambda series: series.fillna(series.median())
            )
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned["Date"] = cleaned["Date"].ffill()

    q1 = cleaned["Sales"].quantile(0.25)
    q3 = cleaned["Sales"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (cleaned["Sales"] < lower_bound) | (cleaned["Sales"] > upper_bound)
    outlier_count = int(outlier_mask.sum())
    cleaned["Sales"] = cleaned["Sales"].clip(lower=lower_bound, upper=upper_bound)

    cleaned = cleaned.sort_values(["Store", "Date"]).reset_index(drop=True)
    summary = {
        "initial_rows": float(initial_rows),
        "rows_after_cleaning": float(len(cleaned)),
        "duplicates_removed": float(duplicate_rows),
        "sales_outliers_capped": float(outlier_count),
        "lower_outlier_bound": float(lower_bound),
        "upper_outlier_bound": float(upper_bound),
    }
    return cleaned, summary


def create_eda_plots(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Path]:
    """Create core EDA charts and save them to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_sales = df.groupby("Date", as_index=False)["Sales"].sum()
    monthly_sales = (
        df.assign(YearMonth=df["Date"].dt.to_period("M").astype(str))
        .groupby("YearMonth", as_index=False)["Sales"]
        .sum()
    )
    month_name_sales = (
        df.assign(MonthName=df["Date"].dt.month_name())
        .groupby("MonthName", as_index=False)["Sales"]
        .sum()
    )
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_name_sales["MonthName"] = pd.Categorical(
        month_name_sales["MonthName"], categories=month_order, ordered=True
    )
    month_name_sales = month_name_sales.sort_values("MonthName")

    yearly_sales = (
        df.assign(Year=df["Date"].dt.year)
        .groupby("Year", as_index=False)["Sales"]
        .sum()
    )
    seasonal_heatmap = (
        df.assign(Year=df["Date"].dt.year, Month=df["Date"].dt.month_name().str[:3])
        .groupby(["Year", "Month"])["Sales"]
        .sum()
        .reset_index()
        .pivot(index="Year", columns="Month", values="Sales")
    )
    seasonal_heatmap = seasonal_heatmap.reindex(columns=[m[:3] for m in month_order])

    saved_paths: Dict[str, Path] = {}

    plt.figure()
    sns.lineplot(data=daily_sales, x="Date", y="Sales", color="#1f77b4")
    plt.title("Daily Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    path = output_dir / "sales_over_time.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["sales_over_time"] = path

    plt.figure()
    sns.lineplot(data=monthly_sales, x="YearMonth", y="Sales", marker="o", color="#ff7f0e")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = output_dir / "monthly_sales_trend.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["monthly_sales_trend"] = path

    plt.figure()
    sns.barplot(data=month_name_sales, x="MonthName", y="Sales", color="#2ca02c")
    plt.title("Total Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = output_dir / "monthly_sales_by_month_name.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["monthly_sales_by_month_name"] = path

    plt.figure()
    sns.barplot(data=yearly_sales, x="Year", y="Sales", color="#d62728")
    plt.title("Yearly Sales Comparison")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.tight_layout()
    path = output_dir / "yearly_sales_comparison.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["yearly_sales_comparison"] = path

    plt.figure(figsize=(12, 5))
    sns.heatmap(seasonal_heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Seasonality Heatmap: Sales by Year and Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    path = output_dir / "seasonality_heatmap.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    saved_paths["seasonality_heatmap"] = path

    return saved_paths


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based, lag-based, and rolling features."""
    featured = df.copy()
    featured["Year"] = featured["Date"].dt.year
    featured["Month"] = featured["Date"].dt.month
    featured["Day"] = featured["Date"].dt.day
    featured["DayOfWeek"] = featured["Date"].dt.dayofweek
    featured["WeekOfYear"] = featured["Date"].dt.isocalendar().week.astype(int)
    featured["IsWeekend"] = (featured["DayOfWeek"] >= 5).astype(int)

    grouped_sales = featured.groupby("Store")["Sales"]
    featured["lag_1"] = grouped_sales.shift(1)
    featured["lag_7"] = grouped_sales.shift(7)
    featured["lag_30"] = grouped_sales.shift(30)
    featured["rolling_mean_7"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(window=7, min_periods=7).mean()
    )
    featured["rolling_mean_30"] = grouped_sales.transform(
        lambda series: series.shift(1).rolling(window=30, min_periods=30).mean()
    )

    featured = featured.dropna(subset=["lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"])
    featured = featured.reset_index(drop=True)
    return featured


def time_based_split(featured: pd.DataFrame, test_days: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use a chronological split with the last N days as test data."""
    ordered_dates = np.array(sorted(featured["Date"].unique()))
    if len(ordered_dates) <= test_days:
        raise ValueError("Not enough dates to create the requested time-based split.")
    split_date = pd.Timestamp(ordered_dates[-test_days])
    train_data = featured[featured["Date"] < split_date].copy()
    test_data = featured[featured["Date"] >= split_date].copy()
    return train_data, test_data


def build_models() -> Dict[str, Pipeline]:
    """Build notebook-friendly yet production-ready ML pipelines."""
    categorical_features = ["Store"]
    numeric_features = [column for column in FEATURE_COLUMNS if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    models = {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    return models


def train_models(train_data: pd.DataFrame) -> Dict[str, Pipeline]:
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["Sales"]

    trained_models = {}
    for model_name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline
    return trained_models


def evaluate_models(
    models: Dict[str, Pipeline], test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["Sales"]

    evaluation_rows = []
    predictions: Dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        prediction = model.predict(X_test)
        predictions[model_name] = prediction
        evaluation_rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, prediction),
                "RMSE": root_mean_squared_error(y_test, prediction),
            }
        )

    metrics = pd.DataFrame(evaluation_rows).sort_values("RMSE").reset_index(drop=True)
    return metrics, predictions


def aggregate_predictions_for_plot(test_data: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    plot_df = test_data[["Date", "Sales"]].copy()
    plot_df["PredictedSales"] = prediction
    plot_df = (
        plot_df.groupby("Date", as_index=False)[["Sales", "PredictedSales"]]
        .sum()
        .rename(columns={"Sales": "ActualSales"})
    )
    return plot_df


def plot_actual_vs_predicted(
    test_data: pd.DataFrame,
    prediction: np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    plot_df = aggregate_predictions_for_plot(test_data, prediction)
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=plot_df, x="Date", y="ActualSales", label="Actual Sales", color="#1f77b4")
    sns.lineplot(data=plot_df, x="Date", y="PredictedSales", label="Predicted Sales", color="#ff7f0e")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def _future_holiday_flag(date_value: pd.Timestamp) -> int:
    known_holidays = {"01-01", "07-04", "11-28", "12-25"}
    return int(date_value.strftime("%m-%d") in known_holidays)


def recursive_forecast(
    model: Pipeline,
    history_df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """Forecast future sales for each store one day at a time using recursive features."""
    forecast_history = history_df.copy()
    forecast_rows = []

    latest_date = forecast_history["Date"].max()
    stores = sorted(forecast_history["Store"].unique().tolist())

    recent_promo_mode = (
        forecast_history.sort_values(["Store", "Date"])
        .groupby("Store")["Promo"]
        .apply(lambda series: int(round(series.tail(30).mean())))
        .to_dict()
    )

    for step in range(1, horizon + 1):
        forecast_date = latest_date + pd.Timedelta(days=step)
        day_rows = []
        for store in stores:
            store_history = forecast_history.loc[forecast_history["Store"] == store].sort_values("Date")
            recent_sales = store_history["Sales"]

            features = {
                "Store": store,
                "Promo": recent_promo_mode.get(store, 0),
                "Holiday": _future_holiday_flag(forecast_date),
                "Year": forecast_date.year,
                "Month": forecast_date.month,
                "Day": forecast_date.day,
                "DayOfWeek": forecast_date.dayofweek,
                "WeekOfYear": int(forecast_date.isocalendar().week),
                "IsWeekend": int(forecast_date.dayofweek >= 5),
                "lag_1": recent_sales.iloc[-1],
                "lag_7": recent_sales.iloc[-7],
                "lag_30": recent_sales.iloc[-30],
                "rolling_mean_7": recent_sales.tail(7).mean(),
                "rolling_mean_30": recent_sales.tail(30).mean(),
            }

            feature_frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            predicted_sales = float(model.predict(feature_frame)[0])
            row = {
                "Date": forecast_date,
                "Store": store,
                "Sales": predicted_sales,
                "Promo": features["Promo"],
                "Holiday": features["Holiday"],
            }
            row.update(features)
            day_rows.append(row)
            forecast_rows.append(row.copy())

        forecast_history = pd.concat([forecast_history, pd.DataFrame(day_rows)], ignore_index=True)

    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df


def plot_future_forecast(forecast_df: pd.DataFrame, output_path: str | Path) -> Path:
    plot_df = forecast_df.groupby("Date", as_index=False)["Sales"].sum()
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=plot_df, x="Date", y="Sales", marker="o", color="#2ca02c")
    plt.title("30-Day Future Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Forecasted Sales")
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_feature_importance(model: Pipeline, output_path: str | Path) -> Path:
    preprocessor = model.named_steps["preprocessor"]
    trained_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": trained_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(12).copy()
    top_features["Feature"] = top_features["Feature"].str.replace("categorical__encoder__", "", regex=False)
    top_features["Feature"] = top_features["Feature"].str.replace("numeric__", "", regex=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_features, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def export_dashboard_data(
    cleaned_df: pd.DataFrame,
    test_data: pd.DataFrame,
    best_model_name: str,
    best_prediction: np.ndarray,
    future_forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Export Power BI / Tableau friendly tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_vs_pred = aggregate_predictions_for_plot(test_data, best_prediction).copy()
    actual_vs_pred["Model"] = best_model_name

    monthly_summary = (
        cleaned_df.assign(
            Year=cleaned_df["Date"].dt.year,
            Month=cleaned_df["Date"].dt.month,
            MonthName=cleaned_df["Date"].dt.month_name(),
        )
        .groupby(["Year", "Month", "MonthName"], as_index=False)["Sales"]
        .sum()
        .sort_values(["Year", "Month"])
    )

    store_summary = (
        cleaned_df.groupby("Store", as_index=False)
        .agg(
            TotalSales=("Sales", "sum"),
            AverageDailySales=("Sales", "mean"),
            PromoDays=("Promo", "sum"),
            HolidayDays=("Holiday", "sum"),
        )
        .sort_values("TotalSales", ascending=False)
    )

    forecast_summary = (
        future_forecast.groupby("Date", as_index=False)["Sales"]
        .sum()
        .rename(columns={"Sales": "ForecastedSales"})
    )

    kpi_summary = pd.DataFrame(
        [
            {
                "Metric": "Best Model",
                "Value": best_model_name,
            },
            {
                "Metric": "Best Model MAE",
                "Value": f"{metrics.iloc[0]['MAE']:.4f}",
            },
            {
                "Metric": "Best Model RMSE",
                "Value": f"{metrics.iloc[0]['RMSE']:.4f}",
            },
            {
                "Metric": "Historical Total Sales",
                "Value": f"{cleaned_df['Sales'].sum():.2f}",
            },
            {
                "Metric": "30 Day Forecast Total",
                "Value": f"{future_forecast['Sales'].sum():.2f}",
            },
            {
                "Metric": "Top Store",
                "Value": str(store_summary.iloc[0]["Store"]),
            },
        ]
    )

    saved = {}
    saved["actual_vs_predicted"] = output_dir / "dashboard_actual_vs_predicted.csv"
    saved["monthly_summary"] = output_dir / "dashboard_monthly_summary.csv"
    saved["store_summary"] = output_dir / "dashboard_store_summary.csv"
    saved["forecast_summary"] = output_dir / "dashboard_forecast_summary.csv"
    saved["kpi_summary"] = output_dir / "dashboard_kpi_summary.csv"
    saved["model_metrics"] = output_dir / "dashboard_model_metrics.csv"

    actual_vs_pred.to_csv(saved["actual_vs_predicted"], index=False)
    monthly_summary.to_csv(saved["monthly_summary"], index=False)
    store_summary.to_csv(saved["store_summary"], index=False)
    forecast_summary.to_csv(saved["forecast_summary"], index=False)
    kpi_summary.to_csv(saved["kpi_summary"], index=False)
    metrics.to_csv(saved["model_metrics"], index=False)
    return saved


def plot_executive_dashboard(
    cleaned_df: pd.DataFrame,
    future_forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create a single dashboard-style image for presentations."""
    daily_sales = cleaned_df.groupby("Date", as_index=False)["Sales"].sum()
    monthly_sales = (
        cleaned_df.assign(YearMonth=cleaned_df["Date"].dt.to_period("M").astype(str))
        .groupby("YearMonth", as_index=False)["Sales"]
        .sum()
    )
    store_sales = (
        cleaned_df.groupby("Store", as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
        .head(5)
    )
    forecast_daily = future_forecast.groupby("Date", as_index=False)["Sales"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("Sales Forecasting Executive Dashboard", fontsize=22, fontweight="bold")

    sns.lineplot(data=daily_sales, x="Date", y="Sales", ax=axes[0, 0], color="#1f77b4")
    axes[0, 0].set_title("Historical Daily Sales")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Sales")

    sns.lineplot(data=monthly_sales, x="YearMonth", y="Sales", ax=axes[0, 1], marker="o", color="#ff7f0e")
    axes[0, 1].set_title("Monthly Sales Trend")
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Sales")
    axes[0, 1].tick_params(axis="x", rotation=45)

    sns.barplot(data=store_sales, x="Sales", y="Store", ax=axes[1, 0], hue="Store", palette="Blues_r", legend=False)
    axes[1, 0].set_title("Top 5 Stores by Total Sales")
    axes[1, 0].set_xlabel("Total Sales")
    axes[1, 0].set_ylabel("Store")

    sns.lineplot(data=forecast_daily, x="Date", y="Sales", ax=axes[1, 1], marker="o", color="#2ca02c")
    axes[1, 1].set_title("Next 30 Days Forecast")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Forecasted Sales")

    best_model = metrics.iloc[0]
    fig.text(
        0.5,
        0.02,
        (
            f"Best model: {best_model['Model']} | "
            f"MAE: {best_model['MAE']:.2f} | RMSE: {best_model['RMSE']:.2f} | "
            f"Historical sales: {cleaned_df['Sales'].sum():,.2f} | "
            f"Forecast total next 30 days: {future_forecast['Sales'].sum():,.2f}"
        ),
        ha="center",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def generate_business_insights(df: pd.DataFrame, forecast_df: pd.DataFrame) -> List[str]:
    daily_sales = df.groupby("Date", as_index=False)["Sales"].sum()
    monthly_sales = (
        df.assign(MonthName=df["Date"].dt.month_name(), Month=df["Date"].dt.month)
        .groupby(["Month", "MonthName"], as_index=False)["Sales"]
        .sum()
        .sort_values("Sales", ascending=False)
    )
    yearly_sales = (
        df.assign(Year=df["Date"].dt.year)
        .groupby("Year", as_index=False)["Sales"]
        .sum()
        .sort_values("Year")
    )
    store_sales = df.groupby("Store", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    promo_impact = df.groupby("Promo", as_index=False)["Sales"].mean()
    holiday_impact = df.groupby("Holiday", as_index=False)["Sales"].mean()
    future_daily = forecast_df.groupby("Date", as_index=False)["Sales"].sum()

    peak_month = monthly_sales.iloc[0]
    best_store = store_sales.iloc[0]

    yoy_growth = np.nan
    if len(yearly_sales) >= 2 and yearly_sales.iloc[0]["Sales"] != 0:
        yoy_growth = (
            (yearly_sales.iloc[-1]["Sales"] - yearly_sales.iloc[0]["Sales"])
            / yearly_sales.iloc[0]["Sales"]
        ) * 100

    promo_lift = np.nan
    if set(promo_impact["Promo"]) == {0, 1}:
        promo_lift = (
            (promo_impact.loc[promo_impact["Promo"] == 1, "Sales"].iloc[0]
            - promo_impact.loc[promo_impact["Promo"] == 0, "Sales"].iloc[0])
            / promo_impact.loc[promo_impact["Promo"] == 0, "Sales"].iloc[0]
        ) * 100

    holiday_lift = np.nan
    if set(holiday_impact["Holiday"]) == {0, 1}:
        holiday_lift = (
            (holiday_impact.loc[holiday_impact["Holiday"] == 1, "Sales"].iloc[0]
            - holiday_impact.loc[holiday_impact["Holiday"] == 0, "Sales"].iloc[0])
            / holiday_impact.loc[holiday_impact["Holiday"] == 0, "Sales"].iloc[0]
        ) * 100

    insight_lines = [
        (
            f"Sales trend: total yearly sales changed by {yoy_growth:.2f}% from "
            f"{int(yearly_sales.iloc[0]['Year'])} to {int(yearly_sales.iloc[-1]['Year'])}, "
            "which signals whether the business is growing or flattening."
        ),
        (
            f"Peak month: {peak_month['MonthName']} generated the highest cumulative sales "
            f"at {peak_month['Sales']:.2f}, so inventory and staffing should be strongest in that window."
        ),
        (
            f"Best-performing store: Store {best_store['Store']} contributed the highest overall sales "
            f"at {best_store['Sales']:.2f}, making it a useful benchmark for other locations."
        ),
        (
            f"Promotion impact: average sales on promo days were {promo_lift:.2f}% higher than non-promo days, "
            "which supports targeted campaigns during slower periods."
        ),
        (
            f"Holiday impact: average sales on holidays were {holiday_lift:.2f}% different from regular days, "
            "so holiday inventory plans should reflect this demand shift."
        ),
        (
            f"Forecast outlook: the next 30 days average {future_daily['Sales'].mean():.2f} sales per day, "
            "which can guide replenishment and short-term revenue targets."
        ),
    ]
    return insight_lines


def save_outputs(
    models: Dict[str, Pipeline],
    metrics: pd.DataFrame,
    future_forecast: pd.DataFrame,
    insights: List[str],
    models_dir: str | Path,
    metrics_dir: str | Path,
) -> None:
    models_dir = Path(models_dir)
    metrics_dir = Path(metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        file_name = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, models_dir / file_name)

    metrics.to_csv(metrics_dir / "model_comparison.csv", index=False)
    future_forecast.to_csv(metrics_dir / "future_30_day_forecast.csv", index=False)
    (metrics_dir / "business_insights.md").write_text(
        "\n".join([f"- {line}" for line in insights]),
        encoding="utf-8",
    )


def run_pipeline(
    csv_path: str | Path,
    figure_dir: str | Path,
    models_dir: str | Path,
    metrics_dir: str | Path,
    dashboard_dir: str | Path | None = None,
    forecast_horizon: int = 30,
) -> ForecastArtifacts:
    set_plot_style()

    raw_df = load_sales_data(csv_path)
    cleaned_df, _ = clean_sales_data(raw_df)
    create_eda_plots(cleaned_df, figure_dir)

    featured_df = engineer_features(cleaned_df)
    train_data, test_data = time_based_split(featured_df, test_days=60)

    trained_models = train_models(train_data)
    metrics, predictions = evaluate_models(trained_models, test_data)
    best_model_name = metrics.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    plot_actual_vs_predicted(
        test_data=test_data,
        prediction=predictions[best_model_name],
        output_path=Path(figure_dir) / "actual_vs_predicted.png",
        title=f"Actual vs Predicted Sales ({best_model_name})",
    )
    plot_future_df = recursive_forecast(best_model, cleaned_df, horizon=forecast_horizon)
    plot_future_forecast(plot_future_df, Path(figure_dir) / "future_forecast.png")

    if "Random Forest" in trained_models:
        plot_feature_importance(
            trained_models["Random Forest"],
            Path(figure_dir) / "random_forest_feature_importance.png",
        )

    dashboard_dir = Path(dashboard_dir) if dashboard_dir is not None else Path(metrics_dir).parent / "dashboard"
    export_dashboard_data(
        cleaned_df=cleaned_df,
        test_data=test_data,
        best_model_name=best_model_name,
        best_prediction=predictions[best_model_name],
        future_forecast=plot_future_df,
        metrics=metrics,
        output_dir=dashboard_dir,
    )
    plot_executive_dashboard(
        cleaned_df=cleaned_df,
        future_forecast=plot_future_df,
        metrics=metrics,
        output_path=dashboard_dir / "executive_dashboard.png",
    )

    business_insights = generate_business_insights(cleaned_df, plot_future_df)
    save_outputs(
        models=trained_models,
        metrics=metrics,
        future_forecast=plot_future_df,
        insights=business_insights,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
    )

    return ForecastArtifacts(
        cleaned_data=cleaned_df,
        featured_data=featured_df,
        train_data=train_data,
        test_data=test_data,
        metrics=metrics,
        predictions=predictions,
        future_forecast=plot_future_df,
        business_insights=business_insights,
    )
