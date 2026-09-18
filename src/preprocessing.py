"""Audited cleaning for GPS trajectories."""

import pandas as pd


def clean_trajectories(df):
    required = [
        "person_id",
        "trajectory_id",
        "latitude",
        "longitude",
        "altitude",
        "timestamp",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df.copy()
    input_count = len(x)

    x["latitude"] = pd.to_numeric(x["latitude"], errors="coerce")
    x["longitude"] = pd.to_numeric(x["longitude"], errors="coerce")
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")

    invalid = (
        ~x["latitude"].between(-90, 90)
        | ~x["longitude"].between(-180, 180)
        | x["timestamp"].isna()
    )

    invalid_count = int(invalid.sum())

    x = x.loc[~invalid].copy()

    before_duplicates = len(x)

    x = (
        x.drop_duplicates(
            subset=[
                "person_id",
                "trajectory_id",
                "timestamp",
                "latitude",
                "longitude",
            ]
        )
        .sort_values(["person_id", "trajectory_id", "timestamp"])
        .reset_index(drop=True)
    )

    duplicate_count = before_duplicates - len(x)

    report = {
        "input": input_count,
        "invalid_coordinates_or_time": invalid_count,
        "duplicates": duplicate_count,
        "output": len(x),
    }

    return x, report