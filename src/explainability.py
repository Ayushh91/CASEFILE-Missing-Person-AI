import pandas as pd
def feature_importance(model) -> pd.DataFrame:
    prep=model.named_steps["preprocessor"]; classifier=model.named_steps["classifier"]
    names=prep.get_feature_names_out(); values=classifier.feature_importances_
    return pd.DataFrame({"feature":names,"importance":values}).sort_values("importance",ascending=False)
def explanation_text(predictions: pd.DataFrame, importance: pd.DataFrame) -> str:
    winner=predictions.iloc[0]; factors=", ".join(importance.head(3).feature.str.replace("num__|cat__","",regex=True))
    return f"Area {winner.area_id} is the highest-probability simulated result ({winner.probability:.1f}%). The model relied most on: {factors}. This is a probabilistic academic output, not a location finding."
