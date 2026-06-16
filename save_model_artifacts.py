"""Train HDRPS logistic regression and save deployment artifacts."""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("Dataset/heart.csv")
ARTIFACT_DIR = Path("model_artifacts")

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].copy()
    y = df["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    log_reg_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(log_reg_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_f1 = cross_val_score(log_reg_pipe, X_train, y_train, cv=cv, scoring="f1")

    log_reg_pipe.fit(X_train, y_train)

    y_pred = log_reg_pipe.predict(X_test)
    y_proba = log_reg_pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(log_reg_pipe.named_steps["model"], ARTIFACT_DIR / "model.pkl")
    joblib.dump(log_reg_pipe.named_steps["scaler"], ARTIFACT_DIR / "scaler.pkl")

    metadata = {
        "n_rows": int(df.shape[0]),
        "n_features": int(len(FEATURE_COLS)),
        "feature_order": FEATURE_COLS,
        "random_state": RANDOM_STATE,
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "test_metrics": {k: float(v) for k, v in metrics.items()},
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    print(f"CV F1-score: {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print("Saved artifacts to model_artifacts/: model.pkl, scaler.pkl, metadata.json")


if __name__ == "__main__":
    main()
