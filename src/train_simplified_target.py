"""
Experiment: Simplify VisitMode into 3 Classes (Fixed)
"""

import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from category_encoders import TargetEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ===================== SIMPLIFY TARGET =====================
print("\nSimplifying VisitMode into 3 classes...")

def simplify_visit_mode(mode):
    if mode == 1:
        return "Business"
    elif mode in [2, 3, 4]:
        return "Leisure"
    elif mode == 5:
        return "Solo"
    else:
        return "Other"

df['VisitMode_Simplified'] = df['VisitMode'].apply(simplify_visit_mode)

print("\nNew Target Distribution:")
print(df['VisitMode_Simplified'].value_counts())

# ===================== ENCODE TARGET TO NUMBERS =====================
le = LabelEncoder()
y = le.fit_transform(df['VisitMode_Simplified'])

print(f"\nEncoded classes: {le.classes_}")

# ===================== PREPARE FEATURES =====================
X = df.drop(columns=['Rating', 'VisitMode', 'VisitMode_Simplified'])

# Target Encoding on features
encoder = TargetEncoder(cols=X.columns.tolist())
X_encoded = encoder.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== TRAIN MODEL =====================
print("\nTraining XGBoost on simplified target...")

model = XGBClassifier(
    n_estimators=174,
    learning_rate=0.196,
    max_depth=7,
    subsample=0.944,
    colsample_bytree=0.825,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print(f"\n✅ Accuracy with 3 Classes: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=le.classes_))

# Save model + encoders
joblib.dump(model, MODELS_DIR / "xgboost_3class_model.joblib")
joblib.dump(encoder, MODELS_DIR / "encoder_3class.joblib")
joblib.dump(le, MODELS_DIR / "label_encoder_3class.joblib")

print("\n✅ Model with simplified target saved!")