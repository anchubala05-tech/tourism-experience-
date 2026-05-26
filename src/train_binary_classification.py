"""
Binary Classification: Business vs Leisure (Remove 'Other' class)
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

# ===================== FILTER DATA =====================
print("\nFiltering data (removing 'Other' class)...")

def simplify_visit_mode(mode):
    if mode == 1:
        return "Business"
    elif mode in [2, 3, 4]:
        return "Leisure"
    else:
        return None   # Will be dropped

df['VisitMode_Binary'] = df['VisitMode'].apply(simplify_visit_mode)

# Remove rows with None (Other class)
df = df.dropna(subset=['VisitMode_Binary'])

print(f"\nData shape after filtering: {df.shape}")
print("\nNew Target Distribution:")
print(df['VisitMode_Binary'].value_counts())

# ===================== PREPARE DATA =====================
X = df.drop(columns=['Rating', 'VisitMode', 'VisitMode_Binary'])
y = df['VisitMode_Binary']

# Encode target to 0 and 1
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded classes: {le.classes_}")   # Should show ['Business' 'Leisure']

# Target Encoding on features
encoder = TargetEncoder(cols=X.columns.tolist())
X_encoded = encoder.fit_transform(X, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===================== TRAIN MODEL =====================
print("\nTraining XGBoost (Binary Classification)...")

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
f1 = f1_score(y_test, preds)

print(f"\n✅ Binary Classification Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=le.classes_))

# Save everything
joblib.dump(model, MODELS_DIR / "xgboost_binary_model.joblib")
joblib.dump(encoder, MODELS_DIR / "encoder_binary.joblib")
joblib.dump(le, MODELS_DIR / "label_encoder_binary.joblib")

print("\n✅ Binary classification model saved!")