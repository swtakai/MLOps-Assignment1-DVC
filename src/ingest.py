"""
Task 1 - Dataset Version 1 (raw)

Reads athletes.csv from the local athletes/ folder
and writes it, unmodified, to data/raw/athletes.csv - the raw
ingestion point that DVC will track as Dataset Version 1.
"""

from pathlib import Path

import pandas as pd

RAW_SOURCE = Path("athletes/athletes.csv")
RAW_OUT = Path("data/raw/athletes.csv")


def main():
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RAW_SOURCE)

    # Write out untouched - this IS Dataset Version 1 (raw)
    data.to_csv(RAW_OUT, index=False)
    print(f"Ingested raw dataset: {data.shape[0]} rows, {data.shape[1]} cols")
    print(f"Written to {RAW_OUT}")


if __name__ == "__main__":
    main()