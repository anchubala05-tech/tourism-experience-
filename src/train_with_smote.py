"""
SMOTE + Merging Rare Classes (Fast Improvement Attempt)
"""

import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"
MODELS_DIR = BASE_DIR / "models"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['Rating', 'VisitMode'])
y = df['VisitMode']

print("\nOriginal VisitMode Distribution:")
print(y.value_counts())

# ===================== MERGE RARE CLASSES =====================
# Merge classes with very low count into "Other" (class 0 in our case)
min_count = 500   # You can change this
value_counts = y.value_counts()
rare_classes = value_counts[value_counts < min_count].index.tolist()

if rare_classes:
    print(f"\nMerging rare classes {rare_classes} into 'Other'")
    y = y.replace(rare_classes, 0)   # Merge into class 0

print("\nAfter Merging:")
print(y.value_counts())

# ===================== TARGET ENCODING =====================
encoder = TargetEncoder(cols=X.columns.tolist())
X_encoded = encoder.fit_transform(X, y)

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== APPLY SMOTE =====================
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {X_train.shape}")
print(f"After SMOTE : {X_train_smote.shape}")

# ===================== TRAIN XGBOOST =====================
print("\nTraining XGBoost with SMOTE data...")

model = XGBClassifier(
    n_estimators=174,
    learning_rate=0.196,
    max_depth=7,
    subsample=0.944,
    colsample_bytree=0.825,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_smote, y_train_smote)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="weighted")

print(f"\n✅ Accuracy after SMOTE + Merging: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# Save
joblib.dump(model, MODELS_DIR / "xgboost_smote_model.joblib")
joblib.dump(encoder, MODELS_DIR / "target_encoder.joblib")

print("\n✅ Model saved as xgboost_smote_model.joblib")