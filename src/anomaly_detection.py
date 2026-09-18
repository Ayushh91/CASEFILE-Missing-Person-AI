"""Isolation Forest anomaly detection for fictional mobility analysis."""

import json

import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import MODELS, SEED


FEATURES = [
    "speed_kmh",
    "distance_km",
    "duration_hours",
    "hour",
    "weekday",
]


def detect_anomalies(
    data: pd.DataFrame,
    contamination: float = 0.08,
):
    """
    Detect statistical movement anomalies using Isolation Forest.

    anomaly_label:
        1  = normal movement
       -1  = statistical anomaly

    anomaly_score:
        Higher values indicate stronger anomaly evidence.
    """

    available = [
        feature
        for feature in FEATURES
        if feature in data.columns
    ]

    if not available:
        raise ValueError(
            "No anomaly-detection features are available."
        )

    X = data[available].copy()

    X = X.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    X = X.fillna(0)

    model = IsolationForest(
        contamination=contamination,
        random_state=SEED,
        n_estimators=200,
    )

    model.fit(X)

    out = data.copy()

    # Isolation Forest:
    #  1  = normal
    # -1  = anomaly
    out["anomaly_label"] = model.predict(X)

    # sklearn score_samples:
    # lower = more anomalous.
    # We invert it so that higher = more anomalous.
    out["anomaly_score"] = -model.score_samples(X)

    # ---------------------------------------------------------
    # Anomaly evaluation statistics
    # ---------------------------------------------------------

    total_records = len(out)

    anomaly_count = int(
        (out["anomaly_label"] == -1).sum()
    )

    normal_count = int(
        (out["anomaly_label"] == 1).sum()
    )

    anomaly_percentage = (
        anomaly_count / total_records * 100
        if total_records
        else 0
    )

    score_summary = (
        out["anomaly_score"]
        .describe()
        .to_dict()
    )

    evaluation = {
        "algorithm": "Isolation Forest",
        "contamination": float(contamination),
        "features_used": available,
        "total_records": int(total_records),
        "normal_records": normal_count,
        "anomalous_records": anomaly_count,
        "anomaly_percentage": float(
            anomaly_percentage
        ),
        "score_summary": {
            key: float(value)
            for key, value in score_summary.items()
        },
        "interpretation": (
            "Anomalies represent statistical deviations "
            "from observed movement patterns. They do not "
            "indicate wrongdoing or suspicious behavior."
        ),
        "threshold_note": (
            "The contamination parameter controls the expected "
            "proportion of observations treated as anomalies. "
            "A value of 0.08 was used for this academic simulation."
        ),
        "false_positive_note": (
            "Without ground-truth anomaly labels, conventional "
            "false-positive and false-negative rates cannot be "
            "reliably calculated. Detected anomalies should "
            "therefore be interpreted as statistical signals."
        ),
    }

    # Save anomaly evaluation separately.
    (MODELS / "anomaly_evaluation.json").write_text(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    return out, model, available
