from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from forecasting_pipeline import run_pipeline


def main() -> None:
    artifacts = run_pipeline(
        csv_path=PROJECT_ROOT / "data" / "raw" / "store_sales.csv",
        figure_dir=PROJECT_ROOT / "outputs" / "figures",
        models_dir=PROJECT_ROOT / "models",
        metrics_dir=PROJECT_ROOT / "outputs" / "metrics",
        dashboard_dir=PROJECT_ROOT / "outputs" / "dashboard",
        forecast_horizon=30,
    )

    print("\nFUTURE_ML_01 pipeline completed successfully.\n")
    print("Model comparison:")
    print(artifacts.metrics.to_string(index=False))
    print("\nTop 5 future forecast rows:")
    print(
        artifacts.future_forecast[
            ["Date", "Store", "Sales"]
        ].head().to_string(index=False)
    )
    print("\nArtifacts saved in:")
    print(f"- Models: {PROJECT_ROOT / 'models'}")
    print(f"- Metrics: {PROJECT_ROOT / 'outputs' / 'metrics'}")
    print(f"- Figures: {PROJECT_ROOT / 'outputs' / 'figures'}")
    print(f"- Dashboard: {PROJECT_ROOT / 'outputs' / 'dashboard'}")


if __name__ == "__main__":
    main()
