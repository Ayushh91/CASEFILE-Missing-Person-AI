"""Search-priority scoring for fictional CASEFILE locations."""

import numpy as np
import pandas as pd


WEIGHTS = {
    "ml_probability_score": 0.30,
    "historical_visit_score": 0.20,
    "route_similarity_score": 0.15,
    "distance_relevance_score": 0.15,
    "time_relevance_score": 0.10,
    "anomaly_evidence_score": 0.10,
}


def _scale(series):
    """Scale values to a 0-100 range."""

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    maximum = series.max()

    if pd.isna(maximum) or maximum <= 0:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series / maximum * 100
    ).clip(0, 100)


def _calculate_time_relevance(
    anomalies,
    area_ids,
    current_hour,
):
    """
    Calculate time relevance from actual historical movement.

    For each area, the score is based on how frequently that area
    was historically visited during the current hour.
    """

    result = pd.Series(
        0.0,
        index=area_ids.index,
    )

    if (
        anomalies is None
        or anomalies.empty
        or "area_id" not in anomalies.columns
        or "timestamp" not in anomalies.columns
    ):
        return result

    history = anomalies.copy()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["timestamp"]
    )

    history = history[
        history["area_id"] >= 0
    ]

    if history.empty:
        return result

    # Count visits to each area during the requested hour.
    hourly = (
        history[
            history["timestamp"].dt.hour
            == int(current_hour)
        ]
        .groupby("area_id")
        .size()
    )

    if hourly.empty:
        return result

    mapped = (
        area_ids
        .map(hourly)
        .fillna(0)
    )

    return _scale(mapped)


def _calculate_anomaly_evidence(
    anomalies,
    area_ids,
):
    """
    Calculate anomaly evidence for each area.

    Areas with a higher proportion of anomalous observations receive
    greater anomaly evidence.
    """

    result = pd.Series(
        0.0,
        index=area_ids.index,
    )

    if (
        anomalies is None
        or anomalies.empty
        or "area_id" not in anomalies.columns
        or "anomaly_label" not in anomalies.columns
    ):
        return result

    history = anomalies[
        anomalies["area_id"] >= 0
    ].copy()

    if history.empty:
        return result

    total_by_area = (
        history
        .groupby("area_id")
        .size()
    )

    anomaly_by_area = (
        history[
            history["anomaly_label"] == -1
        ]
        .groupby("area_id")
        .size()
    )

    # Anomaly rate rather than raw anomaly count.
    anomaly_rate = (
        anomaly_by_area
        / total_by_area
        * 100
    ).fillna(0)

    mapped = (
        area_ids
        .map(anomaly_rate)
        .fillna(0)
    )

    return _scale(mapped)


def rank_areas(
    predictions,
    centers,
    last_lat,
    last_lon,
    route,
    anomalies=None,
    current_hour=None,
):
    """
    Calculate and rank search-priority areas.

    Components:
    - ML probability: 30%
    - Historical visit frequency: 20%
    - Route similarity: 15%
    - Distance relevance: 15%
    - Time relevance: 10%
    - Anomaly evidence: 10%
    """

    required_prediction = {
        "area_id",
        "probability",
    }

    required_centers = {
        "area_id",
        "latitude",
        "longitude",
        "visit_frequency",
    }

    missing_prediction = (
        required_prediction
        - set(predictions.columns)
    )

    if missing_prediction:
        raise ValueError(
            f"Prediction columns missing: "
            f"{missing_prediction}"
        )

    missing_centers = (
        required_centers
        - set(centers.columns)
    )

    if missing_centers:
        raise ValueError(
            f"Center columns missing: "
            f"{missing_centers}"
        )

    out = predictions.copy()

    out["area_id"] = (
        out["area_id"]
        .astype(int)
    )

    out = out.merge(
        centers,
        on="area_id",
        how="left",
    )

    # ---------------------------------------------------------
    # 1. ML probability
    # ---------------------------------------------------------

    out["ml_probability_score"] = (
        pd.to_numeric(
            out["probability"],
            errors="coerce",
        )
        .fillna(0)
        .clip(0, 100)
    )

    # ---------------------------------------------------------
    # 2. Historical visit frequency
    # ---------------------------------------------------------

    out["historical_visit_score"] = _scale(
        out["visit_frequency"]
    )

    # ---------------------------------------------------------
    # 3. Route similarity
    # ---------------------------------------------------------

    route_probabilities = {}

    for item in route or []:

        try:
            area = int(
                item["to_area"]
            )

            probability = float(
                item["probability"]
            )

            route_probabilities[area] = max(
                route_probabilities.get(
                    area,
                    0.0,
                ),
                probability,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    out["route_similarity_score"] = (
        out["area_id"]
        .map(route_probabilities)
        .fillna(0)
        * 100
    ).clip(0, 100)

    # ---------------------------------------------------------
    # 4. Distance relevance
    # ---------------------------------------------------------

    distance = np.sqrt(
        (
            out["latitude"]
            - float(last_lat)
        ) ** 2
        +
        (
            out["longitude"]
            - float(last_lon)
        ) ** 2
    )

    maximum_distance = distance.max()

    if (
        pd.isna(maximum_distance)
        or maximum_distance <= 0
    ):
        out["distance_relevance_score"] = 100.0

    else:
        out["distance_relevance_score"] = (
            (
                1
                - distance
                / maximum_distance
            )
            * 100
        ).clip(0, 100)

    # ---------------------------------------------------------
    # 5. Time relevance
    # ---------------------------------------------------------

    if current_hour is None:
        current_hour = 12

    out["time_relevance_score"] = (
        _calculate_time_relevance(
            anomalies,
            out["area_id"],
            int(current_hour),
        ).to_numpy()
    )

    # ---------------------------------------------------------
    # 6. Anomaly evidence
    # ---------------------------------------------------------

    out["anomaly_evidence_score"] = (
        _calculate_anomaly_evidence(
            anomalies,
            out["area_id"],
        ).to_numpy()
    )

    # ---------------------------------------------------------
    # 7. Weighted priority score
    # ---------------------------------------------------------

    out["priority_score"] = 0.0

    for component, weight in WEIGHTS.items():
        out["priority_score"] += (
            weight
            * out[component]
        )

    out["priority_score"] = (
        out["priority_score"]
        .clip(0, 100)
        .round(2)
    )

    # ---------------------------------------------------------
    # 8. Priority bands
    # ---------------------------------------------------------

    out["priority_label"] = pd.cut(
        out["priority_score"],
        bins=[
            -1,
            30,
            60,
            80,
            100,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    return (
        out
        .sort_values(
            "priority_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )