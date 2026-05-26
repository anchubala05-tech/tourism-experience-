"""
Advanced Training with Target Encoding + LightGBM + CatBoost
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Features and Target
    X = df.drop(columns=['Rating', 'VisitMode'])
    y = df['VisitMode']

    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")

    # ===================== TARGET ENCODING =====================
    print("\nApplying Target Encoding...")
    encoder = TargetEncoder(cols=X.columns.tolist())
    X_encoded = encoder.fit_transform(X, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===================== MODELS =====================
    models = {
        "LightGBM": LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostClassifier(iterations=200, random_state=42, verbose=0),
        "XGBoost": XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    best_acc = 0
    best_model = None
    best_name = ""

    print("\n" + "="*60)
    print("ADVANCED MODEL COMPARISON")
    print("="*60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"{name:15} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Save best model + encoder
    joblib.dump(best_model, MODELS_DIR / "best_advanced_model.joblib")
    joblib.dump(encoder, MODELS_DIR / "target_encoder.joblib")

    print(f"\n✅ Best Model: {best_name} with Accuracy = {best_acc:.4f}")
    print("Models saved in 'models/' folder")

if __name__ == "__main__":
    main()