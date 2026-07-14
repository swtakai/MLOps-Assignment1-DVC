"""
Task 6 - Model Evaluation

Loads a trained model and a test split, computes RMSE, MAE, and R^2, and
writes them to a metrics JSON file (DVC can track these as `metrics` in
dvc.yaml so `dvc metrics diff` works between v1 and v2 runs for Task 8).
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    test_df = pd.read_csv(args.test)

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET]

    preds = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "n_test": len(test_df)}

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R^2:  {r2:.4f}")
    print(f"Metrics written to {args.metrics_out}")


if __name__ == "__main__":
    main()
