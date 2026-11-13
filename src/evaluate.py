"""Actual prediction-based model evaluation."""
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)


def evaluate_models(models, X_test, y_test):
    rows, details = [], {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        })
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        details[name] = {
            "predictions": predictions, "probabilities": probabilities,
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "report": classification_report(y_test, predictions, target_names=["Legitimate", "Phishing"], output_dict=True),
            "fpr": fpr, "tpr": tpr,
        }
    return pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True), details
