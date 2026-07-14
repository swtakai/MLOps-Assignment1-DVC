"""
Task 2 - Feature Engineering

Adds the total_lift feature exactly as specified in the assignment PDF:
    total_lift = deadlift + candj + snatch + backsq

Works on either dataset version (raw or cleaned) - pass --input/--output to
point it at v1 or v2. Rows where any of the four lift columns are missing
will end up with a NaN total_lift; those get dropped later in split.py
since they have no usable label.
"""

import argparse
from pathlib import Path

import pandas as pd


def add_total_lift(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["total_lift"] = (
        data["deadlift"] + data["candj"] + data["snatch"] + data["backsq"]
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = add_total_lift(data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    n_missing = data["total_lift"].isna().sum()
    print(f"Added total_lift to {len(data)} rows ({n_missing} are NaN due to missing lift components)")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
