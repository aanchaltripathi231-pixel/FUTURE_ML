from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "sales_demand_forecasting.ipynb"


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def code_cell(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb["cells"] = [
    markdown_cell(
        "# FUTURE_ML_01: Sales & Demand Forecasting for Businesses\n\n"
        "This notebook demonstrates an end-to-end machine learning workflow for forecasting future sales from historical time-series data.\n\n"
        "Dataset source: Kaggle dataset `abhishekjaiswal4896/store-sales-dataset` copied locally into `data/raw/store_sales.csv`.\n\n"
        "Business objective: forecast sales, understand trends and seasonality, compare baseline and advanced models, and generate practical recommendations for business planning."
    ),
    markdown_cell(
        "## 1. Imports and Project Setup\n\n"
        "We start by importing the project utilities, standard data science libraries, and a few notebook helpers."
    ),
    code_cell(
        "from pathlib import Path\n"
        "import sys\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n\n"
        "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "if str(PROJECT_ROOT / 'src') not in sys.path:\n"
        "    sys.path.append(str(PROJECT_ROOT / 'src'))\n\n"
        "from forecasting_pipeline import (\n"
        "    clean_sales_data,\n"
        "    create_eda_plots,\n"
        "    engineer_features,\n"
        "    evaluate_models,\n"
        "    generate_business_insights,\n"
        "    load_sales_data,\n"
        "    plot_actual_vs_predicted,\n"
        "    plot_feature_importance,\n"
        "    plot_future_forecast,\n"
        "    recursive_forecast,\n"
        "    save_outputs,\n"
        "    set_plot_style,\n"
        "    time_based_split,\n"
        "    train_models,\n"
        ")\n\n"
        "set_plot_style()\n"
        "DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'store_sales.csv'\n"
        "FIGURE_DIR = PROJECT_ROOT / 'outputs' / 'figures'\n"
        "MODELS_DIR = PROJECT_ROOT / 'models'\n"
        "METRICS_DIR = PROJECT_ROOT / 'outputs' / 'metrics'\n"
        "FIGURE_DIR.mkdir(parents=True, exist_ok=True)\n"
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "METRICS_DIR.mkdir(parents=True, exist_ok=True)"
    ),
    markdown_cell(
        "## 2. Data Loading\n\n"
        "The dataset is loaded from CSV, the date column is converted into a datetime format, and the records are sorted chronologically for time-series analysis."
    ),
    code_cell(
        "raw_df = load_sales_data(DATA_PATH)\n"
        "print(f'Raw shape: {raw_df.shape}')\n"
        "display(raw_df.head())\n"
        "display(raw_df.describe(include='all').transpose())"
    ),
    markdown_cell(
        "## 3. Data Cleaning\n\n"
        "Here we remove duplicates, fill missing values, and cap extreme outliers using the IQR rule. This keeps the training signal stable without throwing away potentially useful records."
    ),
    code_cell(
        "cleaned_df, cleaning_summary = clean_sales_data(raw_df)\n"
        "cleaning_summary_df = pd.DataFrame([cleaning_summary])\n"
        "display(cleaning_summary_df)\n"
        "display(cleaned_df.head())"
    ),
    markdown_cell(
        "## 4. Exploratory Data Analysis\n\n"
        "We visualize daily sales, monthly trends, seasonality, and yearly comparisons. These charts help explain when demand rises or falls and whether the business has repeatable patterns."
    ),
    code_cell(
        "eda_paths = create_eda_plots(cleaned_df, FIGURE_DIR)\n"
        "for title, path in eda_paths.items():\n"
        "    display(Markdown(f'### {title.replace(\"_\", \" \").title()}'))\n"
        "    display(Image(filename=str(path)))"
    ),
    markdown_cell(
        "## 5. Feature Engineering\n\n"
        "To turn a time series into a supervised learning problem, we create calendar features, lag features, and rolling averages. Lag features help the model learn short-term memory, while rolling means help it understand smoother demand patterns."
    ),
    code_cell(
        "featured_df = engineer_features(cleaned_df)\n"
        "print(f'Feature-engineered shape: {featured_df.shape}')\n"
        "display(featured_df.head())\n"
        "display(featured_df[['Sales', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']].head())"
    ),
    markdown_cell(
        "## 6. Time-Based Train-Test Split\n\n"
        "We use a chronological split instead of random shuffling so the model is always trained on the past and evaluated on the future. This is the correct way to validate forecasting systems."
    ),
    code_cell(
        "train_df, test_df = time_based_split(featured_df, test_days=60)\n"
        "print(f'Train shape: {train_df.shape}')\n"
        "print(f'Test shape: {test_df.shape}')\n"
        "print(f'Train date range: {train_df[\"Date\"].min().date()} to {train_df[\"Date\"].max().date()}')\n"
        "print(f'Test date range: {test_df[\"Date\"].min().date()} to {test_df[\"Date\"].max().date()}')"
    ),
    markdown_cell(
        "## 7. Model Building\n\n"
        "We train two models:\n\n"
        "- `Linear Regression` as a simple baseline.\n"
        "- `Random Forest Regressor` as a stronger non-linear model.\n\n"
        "The comparison tells us whether the additional model complexity creates a meaningful forecasting improvement."
    ),
    code_cell(
        "trained_models = train_models(train_df)\n"
        "metrics_df, predictions = evaluate_models(trained_models, test_df)\n"
        "display(metrics_df)"
    ),
    markdown_cell(
        "## 8. Evaluation and Visualization\n\n"
        "We compare the models with MAE and RMSE, then plot actual versus predicted sales for the best-performing model."
    ),
    code_cell(
        "best_model_name = metrics_df.iloc[0]['Model']\n"
        "best_model = trained_models[best_model_name]\n"
        "actual_vs_pred_path = plot_actual_vs_predicted(\n"
        "    test_data=test_df,\n"
        "    prediction=predictions[best_model_name],\n"
        "    output_path=FIGURE_DIR / 'actual_vs_predicted.png',\n"
        "    title=f'Actual vs Predicted Sales ({best_model_name})',\n"
        ")\n"
        "display(Image(filename=str(actual_vs_pred_path)))\n\n"
        "rf_importance_path = plot_feature_importance(\n"
        "    trained_models['Random Forest'],\n"
        "    FIGURE_DIR / 'random_forest_feature_importance.png',\n"
        ")\n"
        "display(Image(filename=str(rf_importance_path)))"
    ),
    markdown_cell(
        "## 9. Forecasting the Next 30 Days\n\n"
        "The future forecast is generated recursively. For each new day, the model uses the latest available lag values, including its own previous predictions, to forecast the next step."
    ),
    code_cell(
        "future_forecast_df = recursive_forecast(best_model, cleaned_df, horizon=30)\n"
        "future_forecast_path = plot_future_forecast(future_forecast_df, FIGURE_DIR / 'future_forecast.png')\n"
        "display(future_forecast_df.head())\n"
        "display(Image(filename=str(future_forecast_path)))"
    ),
    markdown_cell(
        "## 10. Business Insights and Recommendations\n\n"
        "A forecasting project is valuable only when it leads to better business decisions. This section summarizes the key patterns we found and translates them into actions a business team can use."
    ),
    code_cell(
        "business_insights = generate_business_insights(cleaned_df, future_forecast_df)\n"
        "for insight in business_insights:\n"
        "    display(Markdown(f'- {insight}'))"
    ),
    markdown_cell(
        "## 11. Saving Models and Outputs\n\n"
        "The final step saves the trained models, evaluation metrics, forecast CSV, and business insights so the project can be reused outside the notebook."
    ),
    code_cell(
        "save_outputs(\n"
        "    models=trained_models,\n"
        "    metrics=metrics_df,\n"
        "    future_forecast=future_forecast_df,\n"
        "    insights=business_insights,\n"
        "    models_dir=MODELS_DIR,\n"
        "    metrics_dir=METRICS_DIR,\n"
        ")\n"
        "display(Markdown('### Saved Files'))\n"
        "display(pd.DataFrame({\n"
        "    'artifact': [\n"
        "        'linear_regression.joblib',\n"
        "        'random_forest.joblib',\n"
        "        'model_comparison.csv',\n"
        "        'future_30_day_forecast.csv',\n"
        "        'business_insights.md',\n"
        "    ],\n"
        "    'path': [\n"
        "        str(MODELS_DIR / 'linear_regression.joblib'),\n"
        "        str(MODELS_DIR / 'random_forest.joblib'),\n"
        "        str(METRICS_DIR / 'model_comparison.csv'),\n"
        "        str(METRICS_DIR / 'future_30_day_forecast.csv'),\n"
        "        str(METRICS_DIR / 'business_insights.md'),\n"
        "    ]\n"
        "}))"
    ),
]

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.13",
    },
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
    nbf.write(nb, file)

print(f"Notebook created at: {NOTEBOOK_PATH}")
