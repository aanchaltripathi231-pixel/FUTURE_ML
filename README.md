# FUTURE_ML

Machine learning project repository focused on practical, end-to-end business use cases.

## Featured Project

### FUTURE_ML_01: Sales & Demand Forecasting for Businesses

This project forecasts future sales from historical time-series data and packages the result in a reviewer-friendly way with:

- a complete notebook
- reusable production-style Python pipeline
- saved model artifacts
- forecast CSV outputs
- business insights
- visual dashboard-ready deliverables

## Quick Review Path

If you are reviewing this repository on GitHub, start here:

1. `FUTURE_ML_01/README.md`
2. `FUTURE_ML_01/outputs/dashboard/executive_dashboard.png`
3. `FUTURE_ML_01/outputs/metrics/model_comparison.csv`
4. `FUTURE_ML_01/outputs/metrics/future_30_day_forecast.csv`
5. `FUTURE_ML_01/notebooks/sales_demand_forecasting.ipynb`

## Main Outcome

- Problem: Sales and demand forecasting
- Dataset source: Kaggle retail sales dataset
- Best model: Random Forest Regressor
- Evaluation: MAE and RMSE on a time-based test split
- Forecast horizon: next 30 days
- Presentation layer: Matplotlib visuals plus Power BI / Tableau ready exports

## Repository Structure

```text
FUTURE_ML/
|-- README.md
|-- FUTURE_ML_01/
    |-- README.md
    |-- notebooks/
    |-- src/
    |-- data/
    |-- outputs/
    |-- models/
```

## Notes

- The repository includes generated outputs so results can be reviewed directly from GitHub.
- The project is designed to be easy to inspect from both GitHub and VS Code.
