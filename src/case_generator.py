"""Creates explicitly fictional case records from movement aggregates."""

import numpy as np
import pandas as pd

from .config import SEED


def generate_cases(
    data: pd.DataFrame,
    n_cases: int = 200,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate fictional missing-person investigation cases.

    The target area is selected from the person's historical movement
    areas while using a balanced quota so that the synthetic dataset
    contains adequate representation of each area.
    """

    if "area_id" not in data:
        raise ValueError("Generate cases after assigning area_id.")

    required = {
        "person_id",
        "timestamp",
        "latitude",
        "longitude",
        "area_id",
        "distance_km",
        "speed_kmh",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    rng = np.random.default_rng(seed)

    eligible = data[data["area_id"] >= 0].copy()

    if eligible.empty:
        raise ValueError("No valid area records available for case generation.")

    people = eligible["person_id"].dropna().unique()

    if len(people) == 0:
        raise ValueError("No valid persons available for case generation.")

    # ---------------------------------------------------------
    # Determine available areas
    # ---------------------------------------------------------
    areas = sorted(
        eligible["area_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if len(areas) < 2:
        raise ValueError("At least two areas are required.")

    # ---------------------------------------------------------
    # Balanced target quotas
    # ---------------------------------------------------------
    base_quota = n_cases // len(areas)
    remainder = n_cases % len(areas)

    target_pool = []

    for index, area in enumerate(areas):
        quota = base_quota + (1 if index < remainder else 0)
        target_pool.extend([area] * quota)

    rng.shuffle(target_pool)

    rows = []

    # ---------------------------------------------------------
    # Generate fictional cases
    # ---------------------------------------------------------
    for i, target in enumerate(target_pool):

        # Select a person who has historically visited the target area.
        target_people = eligible.loc[
            eligible["area_id"] == target,
            "person_id"
        ].dropna().unique()

        if len(target_people) == 0:
            continue

        person = rng.choice(target_people)

        hist = (
            eligible[eligible["person_id"] == person]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if hist.empty:
            continue

        # Select a historical observation as the last known observation.
        if len(hist) > 1:
            start_index = max(1, len(hist) // 2)
            last_index = rng.integers(start_index, len(hist))
        else:
            last_index = 0

        last = hist.iloc[last_index]

        prior = hist[hist["timestamp"] < last["timestamp"]]

        if not prior.empty:
            previous = int(prior.iloc[-1]["area_id"])
        else:
            previous = int(last["area_id"])

        # Historical area frequency.
        area_counts = hist["area_id"].value_counts()

        area_counts = area_counts[
            area_counts.index >= 0
        ]

        if area_counts.empty:
            usual_area = int(last["area_id"])
        else:
            usual_area = int(area_counts.idxmax())

        # Movement statistics.
        distances = (
            hist["distance_km"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        speeds = (
            hist["speed_kmh"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        average_distance = (
            float(distances.mean())
            if not distances.empty
            else 0.0
        )

        average_speed = (
            float(speeds.mean())
            if not speeds.empty
            else 0.0
        )

        # Keep contextual variables realistic rather than completely
        # independent random values.
        hour = int(last["timestamp"].hour)

        day = last["timestamp"].day_name()

        # Synthetic contextual weather.
        weather = rng.choice(
            ["Clear", "Cloudy", "Rain"]
        )

        # Synthetic time since last observation.
        time_since_last_seen = int(
            rng.integers(1, 25)
        )

        rows.append(
            {
                "Case_ID": f"SYN-{i + 1:03d}",
                "Person_ID": f"FICTIONAL_{person}",

                "Age_Group": rng.choice(
                    ["18-25", "26-40", "41-60"]
                ),

                "Gender": "Not used for prediction",

                "Last_Latitude": float(
                    last["latitude"]
                ),

                "Last_Longitude": float(
                    last["longitude"]
                ),

                "Last_Seen_Time": last["timestamp"],

                "Day": day,
                "Hour": hour,
                "Weather": weather,

                "Usual_Area": usual_area,

                "Average_Distance": average_distance,

                "Average_Speed": average_speed,

                "Previous_Area": previous,

                "Time_Since_Last_Seen": time_since_last_seen,

                "Target_Area": int(target),
            }
        )

    cases = pd.DataFrame(rows)

    if cases.empty:
        raise ValueError("Unable to generate synthetic cases.")

    return cases