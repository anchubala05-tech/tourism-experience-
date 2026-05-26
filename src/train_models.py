"""
Model Training with Multiple Models + Best Model Selection
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
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_master_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def prepare_features(df):
    features = []
    
    for col in ["VisitYear", "VisitMonth"]:
        if col in df.columns:
            features.append(col)

    for col in ["Continent", "AttractionType"]:
        possible = [c for c in df.columns if col in c]
        if possible:
            features.append(possible[0])

    # Encode categorical columns
    for col in features:
        if df[col].dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df[features], features

def main():
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_PATH)
    X, feature_names = prepare_features(df)
    print(f"Using features: {feature_names}")

    # ===================== REGRESSION =====================
    print("\n" + "="*60)
    print("REGRESSION TASK: Predicting Rating")
    print("="*60)

    y_reg = df["Rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    best_r2 = -np.inf
    best_reg_model = None
    best_reg_name = ""

    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name:20} → R2: {r2:.4f} | RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_reg_model = model
            best_reg_name = name

    # Save best regression model
    joblib.dump(best_reg_model, MODELS_DIR / "best_regression_model.joblib")
    print(f"\n✅ Best Regression Model: {best_reg_name} (Saved)")

    # ===================== CLASSIFICATION =====================
    print("\n" + "="*60)
    print("CLASSIFICATION TASK: Predicting Visit Mode")
    print("="*60)

    visit_col = [c for c in df.columns if "VisitMode" in c and "Id" not in c][0]
    y_cls = df[visit_col]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_cls_encoded = le.fit_transform(y_cls.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls_encoded, test_size=0.2, random_state=42, stratify=y_cls_encoded
    )

    cls_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    best_acc = 0
    best_cls_model = None
    best_cls_name = ""

    for name, model in cls_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        print(f"{name:20} → Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_cls_model = model
            best_cls_name = name

    # Save best classification model
    joblib.dump(best_cls_model, MODELS_DIR / "best_classification_model.joblib")
    joblib.dump(le, MODELS_DIR / "visitmode_label_encoder.joblib")
    print(f"\n✅ Best Classification Model: {best_cls_name} (Saved)")

    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED WITH MODEL SELECTION")
    print("="*60)

if __name__ == "__main__":
    main()