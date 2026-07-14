"""
Task 3 - Train/Test Split

Splits a features-ready dataset into train/test sets using a fixed random
seed and split ratio. Same seed/ratio is used for both v1 and v2 (pass
different --input/--train-out/--test-out for each) so results are
comparable across dataset versions in Task 8.

Rows with a missing total_lift (the prediction target) are dropped first,
since they can't be used for training or evaluation.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "total_lift"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.input)

    n_before = len(data)
    data = data.dropna(subset=[TARGET])
    n_dropped = n_before - len(data)

    train_df, test_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
    )

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_out).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print(f"Dropped {n_dropped} rows with missing {TARGET}")
    print(f"Split ratio: {1 - args.test_size:.0%} train / {args.test_size:.0%} test, seed={args.seed}")
    print(f"Train: {len(train_df)} rows -> {args.train_out}")
    print(f"Test:  {len(test_df)} rows -> {args.test_out}")


if __name__ == "__main__":
    main()
