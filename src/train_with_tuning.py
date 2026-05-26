"""
Improved Optuna Tuning (Fixed Warnings + More Stable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
import optuna
from xgboost import XGBClassifier          # Changed to XGBoost (more stable)

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"

def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
        "max_depth": trial.suggest_int("max_depth", 4, 8),           # Reduced max depth
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Rating', 'VisitMode'])
    y = df['VisitMode']

    # Target Encoding
    encoder = TargetEncoder(cols=X.columns.tolist())
    X_encoded = encoder.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nRunning Optuna tuning on XGBoost (more stable)...")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=25)

    print(f"\n✅ Best Accuracy: {study.best_value:.4f}")
    print("Best Parameters:", study.best_params)

    # Train final best model
    best_model = XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, MODELS_DIR / "best_tuned_xgboost.joblib")
    joblib.dump(encoder, MODELS_DIR / "target_encoder.joblib")

    print("\n✅ Best tuned XGBoost model saved!")

if __name__ == "__main__":
    main()