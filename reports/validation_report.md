# Validation Report

Dataset: REAL GEOLIFE DATA — 1 users, 100 trajectories (reused processed dataset)

Cleaning: Reused previously cleaned and feature-engineered dataset.

Clustering: Reused previously generated clusters.

Anomaly Evaluation: {'algorithm': 'Isolation Forest', 'contamination': 0.08, 'features_used': ['speed_kmh', 'distance_km', 'duration_hours', 'hour', 'weekday'], 'total_records': 89252, 'normal_records': 82117, 'anomalous_records': 7135, 'anomaly_percentage': 7.994218616949761, 'score_summary': {'count': 89252.0, 'mean': 0.44192471183868043, 'std': 0.06370600475254294, 'min': 0.3715435183288948, '25%': 0.3962403525642374, '50%': 0.4132115545562658, '75%': 0.4750824127405058, 'max': 0.8056842124928354}, 'interpretation': 'Anomalies represent statistical deviations from observed movement patterns. They do not indicate wrongdoing or suspicious behavior.', 'threshold_note': 'The contamination parameter controls the expected proportion of observations treated as anomalies. A value of 0.08 was used for this academic simulation.', 'false_positive_note': 'Without ground-truth anomaly labels, conventional false-positive and false-negative rates cannot be reliably calculated. Detected anomalies should therefore be interpreted as statistical signals.'}

Location Evaluation: {'dataset': {'total_cases': 200, 'training_cases': 150, 'testing_cases': 50, 'target_classes': ['0', '1', '2', '3']}, 'models': {'Random Forest': {'accuracy': 0.28, 'precision': 0.28645238095238096, 'recall': 0.28, 'f1_score': 0.2749591149591149, 'macro_precision': 0.28462301587301586, 'macro_recall': 0.280448717948718, 'macro_f1': 0.27429052429052425, 'top_1_accuracy': 0.3, 'top_3_accuracy': 0.78, 'top_5_accuracy': 1.0, 'confusion_matrix': {'labels': ['0', '1', '2', '3'], 'matrix': [[4, 4, 3, 2], [3, 5, 2, 2], [2, 4, 3, 4], [5, 5, 0, 2]]}}, 'Gradient Boosting': {'accuracy': 0.18, 'precision': 0.1876358543417367, 'recall': 0.18, 'f1_score': 0.18179509379509376, 'macro_precision': 0.18853874883286648, 'macro_recall': 0.1794871794871795, 'macro_f1': 0.18196248196248196, 'top_1_accuracy': 0.18, 'top_3_accuracy': 0.64, 'top_5_accuracy': 1.0, 'confusion_matrix': {'labels': ['0', '1', '2', '3'], 'matrix': [[2, 4, 6, 1], [3, 2, 3, 4], [5, 3, 3, 2], [7, 1, 2, 2]]}}}, 'selected_model': 'Random Forest'}

Number of fictional cases: 200

**Interpretation:** anomalies are statistical deviations, not evidence of wrongdoing. This system is an academic mobility simulation and must not guide real-world investigations.
