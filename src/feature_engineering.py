"""Movement features; distance is Haversine kilometres."""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371.0088 * 2 * np.arcsin(np.sqrt(value))


def add_movement_features(
    data: pd.DataFrame,
    max_speed_kmh: float = 250,
) -> pd.DataFrame:

    df = data.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.sort_values(
        ["person_id", "trajectory_id", "timestamp"]
    ).reset_index(drop=True)

    # Required time features
    df["day"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    group = df.groupby(
        ["person_id", "trajectory_id"],
        sort=False,
    )

    previous_latitude = group["latitude"].shift()
    previous_longitude = group["longitude"].shift()
    previous_time = group["timestamp"].shift()

    df["distance_km"] = haversine_km(
        previous_latitude,
        previous_longitude,
        df["latitude"],
        df["longitude"],
    )

    df["duration_hours"] = (
        df["timestamp"] - previous_time
    ).dt.total_seconds() / 3600

    df["speed_kmh"] = np.where(
        df["duration_hours"] > 0,
        df["distance_km"] / df["duration_hours"],
        np.nan,
    )

    # Remove physically unrealistic speeds
    invalid_speed = (
        (df["speed_kmh"] < 0)
        | (df["speed_kmh"] > max_speed_kmh)
    )

    df.loc[invalid_speed, "speed_kmh"] = np.nan

    movement_columns = [
        "distance_km",
        "duration_hours",
        "speed_kmh",
    ]

    df[movement_columns] = (
        df[movement_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Daily distance per person
    df["daily_distance_km"] = (
        df.groupby(["person_id", "date"])["distance_km"]
        .transform("sum")
    )

    # Average movement statistics per person
    df["average_speed_kmh"] = (
        df.groupby("person_id")["speed_kmh"]
        .transform("mean")
    )

    df["average_distance_km"] = (
        df.groupby("person_id")["distance_km"]
        .transform("mean")
    )

    return df
