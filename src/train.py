"""Train and persist the three required phishing classifiers."""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def train_models(X_train, y_train):
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def save_models(models, feature_names, constant_features, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, directory / f"{name.lower().replace(' ', '_')}.joblib")
    joblib.dump({"features": list(feature_names), "constant_features": constant_features}, directory / "metadata.joblib")
