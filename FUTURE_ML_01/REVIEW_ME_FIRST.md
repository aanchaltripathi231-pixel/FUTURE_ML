# REVIEW ME FIRST

If you are reviewing this project quickly, use this order:

1. Open `README.md`
2. Open `outputs/dashboard/executive_dashboard.png`
3. Open `outputs/metrics/model_comparison.csv`
4. Open `outputs/metrics/future_30_day_forecast.csv`
5. Open `notebooks/sales_demand_forecasting.ipynb`
6. Open `src/forecasting_pipeline.py`

## What To Look For

- clear business problem
- chronological train-test split for forecasting
- comparison between baseline and advanced models
- future 30-day forecast output
- visual business presentation layer
- reusable code structure

## Best Model Summary

- Best model: Random Forest
- MAE: 5.1079
- RMSE: 6.4132

## Business Highlights

- sales grew from 2022 to 2023
- December is the strongest month
- Store 9 is the top-performing store
- promotions improve average sales

## Run Commands

```powershell
python run_project.py
python test_project.py
```
