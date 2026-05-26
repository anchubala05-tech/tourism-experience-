"""
Final Experiment: Remove VisitYear + New City Features + CatBoost
"""

import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ===================== FEATURE ENGINEERING =====================
print("\nApplying Feature Engineering...")

# 1. Remove VisitYear (lowest importance)
df = df.drop(columns=['VisitYear'], errors='ignore')

# 2. Create new features from CityName
city_counts = df['CityName'].value_counts()
df['City_Frequency'] = df['CityName'].map(city_counts)

# Group rare cities (< 50 occurrences) into "Rare_City"
df['City_Grouped'] = df['CityName'].apply(
    lambda x: x if city_counts.get(x, 0) >= 50 else 'Rare_City'
)

print("New features created: City_Frequency, City_Grouped")
print("Removed: VisitYear")

# ===================== PREPARE DATA =====================
X = df.drop(columns=['Rating', 'VisitMode'])
y = df['VisitMode']

print(f"\nFinal Features: {list(X.columns)}")

# Target Encoding
encoder = TargetEncoder(cols=X.columns.tolist())
X_encoded = encoder.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== MODELS =====================
print("\n" + "="*60)
print("COMPARING XGBOOST vs CATBOOST")
print("="*60)

models = {
    "XGBoost": XGBClassifier(
        n_estimators=174, learning_rate=0.196, max_depth=7,
        subsample=0.944, colsample_bytree=0.825,
        random_state=42, n_jobs=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200, learning_rate=0.1, depth=7,
        random_state=42, verbose=0
    )
}

best_acc = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    
    print(f"{name:10} → Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print(f"\n✅ Best Model: {best_model_name} with Accuracy = {best_acc:.4f}")

# Save best model
joblib.dump(best_model, MODELS_DIR / "best_final_model.joblib")
joblib.dump(encoder, MODELS_DIR / "final_encoder.joblib")

print("✅ Best model saved!")