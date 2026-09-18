# CASEFILE investigation-support simulation report

## Abstract
This project demonstrates a modular, reproducible ML workflow for fictional cases. It does not locate people.

## Introduction, problem, and objectives
Historical trajectories can be summarized into visited areas and transitions. The objective is a probabilistic ranked-area prototype, including anomaly detection, explainability, mapping, and clear limitations.

## Dataset and preprocessing
GeoLife is the intended public source; see `data_source_document.md`. The loader validates actual file structure. Cleaning coerces coordinates/timestamps, rejects invalid coordinates, removes duplicates, sorts trajectories, derives calendar fields, and records removals.

## Methodology and algorithms
Haversine distance, duration and speed create movement features. K-Means identifies practical area states; DBSCAN is available for comparison and records noise. Isolation Forest identifies statistical deviations. A leakage-safe Random Forest pipeline predicts synthetic `Target_Area`; target is excluded from inputs. Markov transition probabilities produce routes. Priority scoring uses documented weighted, 0–100 components.

## Evaluation and results
Run `python -m src.pipeline`. Computed classification metrics are saved to `models/evaluation.json` and validation details to `reports/validation_report.md`. No results are stated here before execution.

## Explainability
Random Forest importances and score components explain outputs. Feature importance is associative, not causal.

## Limitations, ethics, future scope, conclusion
Outputs are probabilistic and are not real-world search instructions. Fictional identities only; no PII or private data. Bias, false positives, false negatives, and geographic sampling limits are expected. Future work: carefully governed public data and stronger group validation.

## References
Microsoft Research, GeoLife GPS Trajectory Dataset User Guide; scikit-learn documentation; Streamlit documentation.
