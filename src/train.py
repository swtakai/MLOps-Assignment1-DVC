"""
Task 5 - Baseline Machine Learning Model

Trains a baseline Linear Regression model to predict total_lift.

Feature choice: age, height, weight, gender, background, experience,
schedule, howlong, eat, region. The four lift columns (deadlift, candj,
snatch, backsq) are deliberately EXCLUDED even though they're available -
total_lift is literally their sum, so including them would let the model
"cheat" (near-perfect R^2 with zero real predictive insight). The model
instead predicts total_lift from demographic/survey attributes, which is
the more realistic and useful baseline.

This same script is used unchanged for both v1 (raw) and v2 (cleaned) in
Task 8 - just pass different --train paths. A ColumnTransformer handles
missing values and unseen categories so it works whether or not the data
has been cleaned first.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "total_lift"
NUMERIC_FEATURES = ["age", "height", "weight"]
CATEGORICAL_FEATURES = [
    "gender", "background", "experience", "schedule", "howlong", "eat", "region",
]
SEED = 42


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df[TARGET]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    print(f"Trained on {len(train_df)} rows")
    print(f"Features: {NUMERIC_FEATURES + CATEGORICAL_FEATURES}")
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
