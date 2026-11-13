"""Dataset loading and leakage-safe preparation helpers."""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN


def load_data(path):
    """Load the unmodified phishing dataset and validate required columns."""
    data = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data


def dataset_summary(data):
    return {
        "rows": len(data), "columns": len(data.columns),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "class_counts": data[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }


def prepare_features(data):
    """Remove identifier and non-informative constants; keep target separate."""
    X = data.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore").copy()
    y = data[TARGET_COLUMN].copy()
    constant_features = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_features)
    return X, y, constant_features


def make_split(data):
    X, y, constant_features = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, constant_features
