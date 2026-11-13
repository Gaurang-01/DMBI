"""Prediction and batch-upload validation helpers."""
import pandas as pd


def predict_record(model, features):
    frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])
    probability = float(model.predict_proba(frame)[:, 1][0])
    label = int(model.predict(frame)[0])
    risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.35 else "Low"
    return label, probability, risk


def validate_batch(data, feature_names):
    supplied = set(data.columns)
    expected = set(feature_names)
    missing = [c for c in feature_names if c not in supplied]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    return data.loc[:, feature_names].copy()


def batch_predict(model, data, feature_names):
    X = validate_batch(data, feature_names)
    result = data.copy()
    result["prediction"] = model.predict(X)
    result["prediction_label"] = result["prediction"].map({0: "Legitimate", 1: "Phishing"})
    result["phishing_probability"] = model.predict_proba(X)[:, 1]
    result["risk_category"] = pd.cut(result["phishing_probability"], [-0.01, .35, .70, 1], labels=["Low", "Medium", "High"])
    return result
