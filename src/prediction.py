"""Leakage-safe probability models for fictional target areas."""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MODELS, SEED


FEATURES = [
    "Age_Group",
    "Day",
    "Hour",
    "Weather",
    "Last_Latitude",
    "Last_Longitude",
    "Usual_Area",
    "Average_Distance",
    "Average_Speed",
    "Previous_Area",
    "Time_Since_Last_Seen",
]

CATEGORICAL = [
    "Age_Group",
    "Day",
    "Weather",
    "Usual_Area",
    "Previous_Area",
]


def _build_preprocessor():
    """Create the preprocessing pipeline."""

    numeric = [
        feature
        for feature in FEATURES
        if feature not in CATEGORICAL
    ]

    return ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def _build_random_forest():
    """Create Random Forest pipeline."""

    return Pipeline(
        [
            (
                "preprocessor",
                _build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _build_gradient_boosting():
    """Create Gradient Boosting pipeline."""

    return Pipeline(
        [
            (
                "preprocessor",
                _build_preprocessor(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _top_k_accuracy(model, X, y, k):
    """Calculate Top-K classification accuracy."""

    probabilities = model.predict_proba(X)
    classes = model.named_steps["classifier"].classes_

    top_k = min(k, len(classes))

    correct = 0

    for actual, row in zip(y, probabilities):

        top_classes = classes[
            np.argsort(row)[-top_k:]
        ]

        if actual in top_classes:
            correct += 1

    return float(correct / len(y))


def _evaluate_model(model, X_test, y_test):
    """Calculate complete classification metrics."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "top_1_accuracy": _top_k_accuracy(
            model,
            X_test,
            y_test,
            1,
        ),
        "top_3_accuracy": _top_k_accuracy(
            model,
            X_test,
            y_test,
            3,
        ),
        "top_5_accuracy": _top_k_accuracy(
            model,
            X_test,
            y_test,
            5,
        ),
    }

    labels = sorted(
        pd.Series(y_test)
        .astype(str)
        .unique()
        .tolist()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    metrics["confusion_matrix"] = {
        "labels": labels,
        "matrix": matrix.tolist(),
    }

    return metrics


def train_location_model(cases: pd.DataFrame):
    """
    Train and compare location prediction models.

    Random Forest is retained as the final deployed model.
    """

    missing = (
        set(FEATURES + ["Target_Area"])
        - set(cases.columns)
    )

    if missing:
        raise ValueError(
            f"Case columns missing: {missing}"
        )

    X = cases[FEATURES].copy()

    y = cases["Target_Area"].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "At least two target areas are required."
        )

    # Ensure every class has enough observations for stratification.
    stratify = (
        y
        if y.value_counts().min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=stratify,
    )

    models = {
        "Random Forest": _build_random_forest(),
        "Gradient Boosting": _build_gradient_boosting(),
    }

    evaluations = {}

    trained_models = {}

    for name, model in models.items():

        print(f"Evaluating {name}...")

        model.fit(
            X_train,
            y_train,
        )

        evaluations[name] = _evaluate_model(
            model,
            X_test,
            y_test,
        )

        trained_models[name] = model

    # Random Forest remains the deployed model.
    final_model = trained_models["Random Forest"]

    joblib.dump(
        final_model,
        MODELS / "location_model.joblib",
    )

    evaluation = {
        "dataset": {
            "total_cases": int(len(cases)),
            "training_cases": int(len(X_train)),
            "testing_cases": int(len(X_test)),
            "target_classes": sorted(
                y.unique().tolist()
            ),
        },
        "models": evaluations,
        "selected_model": "Random Forest",
    }

    (MODELS / "evaluation.json").write_text(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    # Preserve the original function contract.
    return final_model, evaluation


def predict_areas(
    model,
    case: pd.Series | dict,
) -> pd.DataFrame:
    """Predict probability for every target area."""

    frame = pd.DataFrame(
        [dict(case)]
    )[FEATURES]

    probabilities = model.predict_proba(frame)[0]

    classes = (
        model
        .named_steps["classifier"]
        .classes_
    )

    result = pd.DataFrame(
        {
            "area_id": classes,
            "probability": probabilities * 100,
        }
    )

    result = (
        result
        .sort_values(
            "probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if not np.isclose(
        result["probability"].sum(),
        100,
    ):
        raise RuntimeError(
            "Invalid probability output"
        )

    return result


def load_location_model():
    """Load the trained location model."""

    return joblib.load(
        MODELS / "location_model.joblib"
    )