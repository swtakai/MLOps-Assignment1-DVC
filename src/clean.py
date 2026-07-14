"""
Task 1 - Dataset Version 2 (cleaned/processed)

Applies the cleaning logic exactly as specified in the assignment PDF to
data/raw/athletes.csv and writes the result to data/processed/athletes_clean.csv.
This is Dataset Version 2 that DVC will track.

NOTE: The cleaning block below is copied verbatim from the assignment
instructions (Assignment #1, "Clean data using following code"). It is not
rewritten or reorganized so that grading can map it directly back to the PDF.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_IN = Path("data/raw/athletes.csv")
PROCESSED_OUT = Path("data/processed/athletes_clean.csv")


def main():
    PROCESSED_OUT.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(RAW_IN)
    n_raw = len(data)

    # Cleaning logic copied exactly from the assignment PDF

    # Remove non-relevant columns
    data = data.dropna(
        subset=[
            'region', 'age', 'weight', 'height', 'howlong',
            'gender', 'eat', 'background', 'experience',
            'schedule', 'deadlift', 'candj', 'snatch', 'backsq'
        ]
    )

    data = data.drop(
        columns=[
            'affiliate', 'team', 'name', 'athlete_id',
            'fran', 'helen', 'grace', 'filthy50',
            'fgonebad', 'run400', 'run5k', 'pullups', 'train'
        ]
    )

    # Remove outliers
    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]

    data = data[
        ((data['gender'] == 'Male') & (data['deadlift'] <= 1105)) |
        ((data['gender'] == 'Female') & (data['deadlift'] <= 636))
    ]

    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

    # Clean survey data
    decline_dict = {'Decline to answer|': np.nan}
    data = data.replace(decline_dict)

    data = data.dropna(
        subset=[
            'background', 'experience',
            'schedule', 'howlong', 'eat'
        ]
    )

    data.to_csv(PROCESSED_OUT, index=False)
    print(f"Raw rows:       {n_raw}")
    print(f"Cleaned rows:   {len(data)} ({len(data)/n_raw:.1%} retained)")
    print(f"Cleaned cols:   {data.shape[1]}")
    print(f"Written to {PROCESSED_OUT}")


if __name__ == "__main__":
    main()