"""
Improved Feature Engineering
Run this to create a better dataset for training
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).parent.parent
CLEANED_PATH = BASE_DIR / "data" / "processed" / "cleaned_master_data.csv"
FINAL_DATA_PATH = BASE_DIR / "data" / "processed" / "final_model_data.csv"

print("Loading cleaned data...")
df = pd.read_csv(CLEANED_PATH)
print(f"Original shape: {df.shape}")

# ===================== SELECT BETTER FEATURES =====================
selected_cols = [
    'VisitYear', 'VisitMonth', 'Rating', 
    'Continent', 'Country', 'CityName',           # Location features
    'Attraction', 'AttractionType',               # Attraction features
    'VisitMode'                                   # Target for classification
]

# Keep only columns that exist
available_cols = [col for col in selected_cols if col in df.columns]
df = df[available_cols].copy()

print(f"Using columns: {available_cols}")

# ===================== ENCODE CATEGORICAL COLUMNS =====================
categorical_cols = ['Continent', 'Country', 'CityName', 'Attraction', 'AttractionType', 'VisitMode']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# ===================== SAVE FINAL DATA =====================
df.to_csv(FINAL_DATA_PATH, index=False)
print(f"\n✅ Final model data saved: {FINAL_DATA_PATH}")
print(f"Final shape: {df.shape}")
print("\nSample of final data:")
print(df.head())