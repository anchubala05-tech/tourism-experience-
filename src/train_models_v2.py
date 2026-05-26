"""
Improved Model Training with Better Features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

from xgboost import XGBRegressor, XGBClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    print("Loading improved data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")

    # ===================== FEATURES & TARGET =====================
    # Drop target columns from features
    X = df.drop(columns=['Rating', 'VisitMode'], errors='ignore')
    y_reg = df['Rating']
    y_cls = df['VisitMode']

    print(f"Features used: {list(X.columns)}")

    # ===================== REGRESSION =====================
    print("\n" + "="*60)
    print("REGRESSION: Predicting Rating")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    }

    best_r2 = -np.inf
    best_reg_name = ""
    best_reg_model = None

    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name:20} → R2: {r2:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_reg_name = name
            best_reg_model = model

    joblib.dump(best_reg_model, MODELS_DIR / "best_regression_model.joblib")
    print(f"\n✅ Best Regression Model: {best_reg_name}")

    # ===================== CLASSIFICATION =====================
    print("\n" + "="*60)
    print("CLASSIFICATION: Predicting Visit Mode")
    print("="*60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    cls_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    }

    best_acc = 0
    best_cls_name = ""
    best_cls_model = None

    for name, model in cls_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"{name:20} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_cls_name = name
            best_cls_model = model

    joblib.dump(best_cls_model, MODELS_DIR / "best_classification_model.joblib")
    print(f"\n✅ Best Classification Model: {best_cls_name}")

    print("\n" + "="*60)
    print("🎉 IMPROVED TRAINING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()