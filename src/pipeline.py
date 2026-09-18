"""Run the CASEFILE pipeline efficiently using existing processed data."""

import json
import joblib
import pandas as pd

from .config import PROCESSED, SYNTHETIC, MODELS, REPORTS
from .data_loader import load_source
from .preprocessing import clean_trajectories
from .feature_engineering import add_movement_features
from .clustering import cluster_locations
from .anomaly_detection import detect_anomalies
from .case_generator import generate_cases
from .prediction import train_location_model
from .route_prediction import build_transition_matrix


def create_anomaly_evaluation(data):
    """Create evaluation information from existing anomaly results."""

    if "anomaly_label" not in data.columns:
        raise ValueError(
            "Existing processed data does not contain anomaly labels."
        )

    total_records = len(data)

    anomalous_records = int(
        (data["anomaly_label"] == -1).sum()
    )

    normal_records = int(
        (data["anomaly_label"] == 1).sum()
    )

    anomaly_percentage = (
        anomalous_records / total_records * 100
        if total_records
        else 0
    )

    evaluation = {
        "algorithm": "Isolation Forest",
        "contamination": 0.08,
        "features_used": [
            "speed_kmh",
            "distance_km",
            "duration_hours",
            "hour",
            "weekday",
        ],
        "total_records": int(total_records),
        "normal_records": normal_records,
        "anomalous_records": anomalous_records,
        "anomaly_percentage": float(anomaly_percentage),
        "score_summary": {},
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

    if "anomaly_score" in data.columns:
        summary = data["anomaly_score"].describe().to_dict()

        evaluation["score_summary"] = {
            key: float(value)
            for key, value in summary.items()
        }

    (MODELS / "anomaly_evaluation.json").write_text(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    return evaluation


def run_pipeline():

    processed_file = PROCESSED / "movement_features.csv"
    centers_file = PROCESSED / "area_centers.csv"

    # ---------------------------------------------------------
    # STEP 1: Reuse existing processed GeoLife data
    # ---------------------------------------------------------

    if processed_file.exists() and centers_file.exists():

        print("Existing processed dataset found.")
        print("Reusing movement_features.csv...")

        anomalous = pd.read_csv(
            processed_file,
            parse_dates=["timestamp"],
        )

        centers = pd.read_csv(
            centers_file
        )

        print(
            f"Loaded {len(anomalous):,} "
            "processed movement records."
        )

        status = (
            "REAL GEOLIFE DATA — "
            f"{anomalous['person_id'].nunique()} users, "
            f"{anomalous['trajectory_id'].nunique()} trajectories "
            "(reused processed dataset)"
        )

        cluster_model = None
        anomaly_model = None

        cluster_model_file = (
            MODELS / "clustering_model.joblib"
        )

        anomaly_model_file = (
            MODELS / "anomaly_model.joblib"
        )

        if cluster_model_file.exists():
            cluster_model = joblib.load(
                cluster_model_file
            )

        if anomaly_model_file.exists():
            anomaly_model = joblib.load(
                anomaly_model_file
            )

        clean_report = (
            "Reused previously cleaned and "
            "feature-engineered dataset."
        )

        cluster_metrics = (
            "Reused previously generated clusters."
        )

        # Create anomaly evaluation from the existing
        # anomaly labels instead of rerunning Isolation Forest.
        print("Creating anomaly evaluation...")

        anomaly_metrics = create_anomaly_evaluation(
            anomalous
        )

    # ---------------------------------------------------------
    # STEP 2: Full processing if processed data is absent
    # ---------------------------------------------------------

    else:

        print(
            "No processed dataset found."
        )

        print(
            "Running full GeoLife processing pipeline..."
        )

        raw, status = load_source()

        clean, clean_report = (
            clean_trajectories(raw)
        )

        features = add_movement_features(
            clean
        )

        (
            clustered,
            cluster_model,
            centers,
            cluster_metrics,
        ) = cluster_locations(
            features
        )

        (
            anomalous,
            anomaly_model,
            _,
        ) = detect_anomalies(
            clustered
        )

        anomaly_metrics = json.loads(
            (
                MODELS /
                "anomaly_evaluation.json"
            ).read_text()
        )

    # ---------------------------------------------------------
    # STEP 3: Generate fictional cases
    # ---------------------------------------------------------

    print("Generating fictional cases...")

    cases = generate_cases(
        anomalous,
        n_cases=200,
    )

    print(
        f"Generated {len(cases)} fictional cases."
    )

    # ---------------------------------------------------------
    # STEP 4: Train location prediction models
    # ---------------------------------------------------------

    print(
        "Training location prediction model..."
    )

    model, metrics = train_location_model(
        cases
    )

    print(
        "Location model training completed."
    )

    # ---------------------------------------------------------
    # STEP 5: Route prediction
    # ---------------------------------------------------------

    print(
        "Building route transition matrix..."
    )

    transitions = build_transition_matrix(
        anomalous
    )

    # ---------------------------------------------------------
    # STEP 6: Create directories
    # ---------------------------------------------------------

    for path in [
        PROCESSED,
        SYNTHETIC,
        MODELS,
        REPORTS,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # STEP 7: Save outputs
    # ---------------------------------------------------------

    anomalous.to_csv(
        processed_file,
        index=False,
    )

    centers.to_csv(
        centers_file,
        index=False,
    )

    cases.to_csv(
        SYNTHETIC / "fictional_cases.csv",
        index=False,
    )

    if cluster_model is not None:
        joblib.dump(
            cluster_model,
            MODELS / "clustering_model.joblib",
        )

    if anomaly_model is not None:
        joblib.dump(
            anomaly_model,
            MODELS / "anomaly_model.joblib",
        )

    (
        MODELS / "transitions.json"
    ).write_text(
        json.dumps(
            transitions,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # STEP 8: Validation report
    # ---------------------------------------------------------

    (
        REPORTS / "validation_report.md"
    ).write_text(
        f"# Validation Report\n\n"
        f"Dataset: {status}\n\n"
        f"Cleaning: {clean_report}\n\n"
        f"Clustering: {cluster_metrics}\n\n"
        f"Anomaly Evaluation: {anomaly_metrics}\n\n"
        f"Location Evaluation: {metrics}\n\n"
        f"Number of fictional cases: "
        f"{len(cases)}\n\n"
        f"**Interpretation:** anomalies are statistical "
        f"deviations, not evidence of wrongdoing. "
        f"This system is an academic mobility simulation "
        f"and must not guide real-world investigations.\n"
    )

    return {
        "dataset": status,
        "cleaning": clean_report,
        "clusters": cluster_metrics,
        "anomalies": anomaly_metrics,
        "metrics": metrics,
        "cases": len(cases),
    }


if __name__ == "__main__":

    print(
        json.dumps(
            run_pipeline(),
            indent=2,
            default=str,
        )
    )