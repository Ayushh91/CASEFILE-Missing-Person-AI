
"""Robust GeoLife loader with explicit real-data validation and demo fallback."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_GEOLIFE, SEED


COLUMNS = [
    "latitude",
    "longitude",
    "code",
    "altitude",
    "date_days",
    "date",
    "time",
]

EMPTY = [
    "person_id",
    "trajectory_id",
    "latitude",
    "longitude",
    "altitude",
    "timestamp",
]


def geolife_files(root=RAW_GEOLIFE):
    """Return all GeoLife .plt files from the dataset directory."""
    root = Path(root)

    if not root.exists():
        return []

    return sorted(root.rglob("*.plt"))


def load_geolife(root=RAW_GEOLIFE, max_files=500):
    """
    Load GeoLife trajectory files.

    max_files=500 processes the first 500 files.
    Use max_files=None to process the complete dataset.
    """
    rows = []

    files = geolife_files(root)
    files = files[:max_files]

    print(f"Found {len(files)} GeoLife files to process.")

    for index, path in enumerate(files, start=1):
        try:
            df = pd.read_csv(
                path,
                skiprows=6,
                names=COLUMNS,
                header=None,
            )

            df["timestamp"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
            )

            person_id = path.parent.parent.name

            df["person_id"] = person_id
            df["trajectory_id"] = path.stem

            selected = df[
                [
                    "person_id",
                    "trajectory_id",
                    "latitude",
                    "longitude",
                    "altitude",
                    "timestamp",
                ]
            ]

            rows.append(selected)

            if index % 25 == 0 or index == len(files):
                print(f"Processed {index}/{len(files)} files.")

        except Exception as exc:
            raise ValueError(
                f"Unable to parse {path}: {exc}"
            ) from exc

    if not rows:
        return pd.DataFrame(columns=EMPTY)

    return pd.concat(rows, ignore_index=True)


def generate_demo_trajectories(
    n_people=8,
    points_per_person=120,
    seed=SEED,
):
    """Generate synthetic trajectories when real data is unavailable."""
    rng = np.random.default_rng(seed)

    centers = np.array(
        [
            [39.9042, 116.4074],
            [39.914, 116.387],
            [39.884, 116.427],
            [39.925, 116.435],
        ]
    )

    output = []

    for person in range(n_people):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=person)

        states = rng.integers(
            0,
            len(centers),
            points_per_person,
        )

        for i, state in enumerate(states):
            lat, lon = centers[state] + rng.normal(0, 0.003, 2)

            output.append(
                {
                    "person_id": f"DEMO_{person:02d}",
                    "trajectory_id": f"demo_{person:02d}",
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": rng.uniform(30, 80),
                    "timestamp": start + pd.Timedelta(minutes=15 * i),
                }
            )

    return pd.DataFrame(output)


def load_source():
    """Load real GeoLife data or use synthetic demo data."""
    data = load_geolife()

    if data.empty:
        return (
            generate_demo_trajectories(),
            "SYNTHETIC DEMO — place GeoLife .plt files in data/raw/GeoLife",
        )

    return (
        data,
        f"REAL GEOLIFE DATA — "
        f"{data['person_id'].nunique()} users, "
        f"{data['trajectory_id'].nunique()} trajectories",
    )