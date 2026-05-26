"""
Data Cleaning Script for Tourism Project
Run this after EDA
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "processed" / "master_data.csv"
CLEANED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_master_data.csv"

print("Loading master data...")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Original Shape: {df.shape}")

# ===================== 1. CHECK MISSING VALUES =====================
print("\n" + "="*60)
print("MISSING VALUES BEFORE CLEANING")
print("="*60)
print(df.isnull().sum()[df.isnull().sum() > 0])

# ===================== 2. HANDLE MISSING VALUES =====================

# Drop rows where critical columns are missing
critical_cols = ['UserId', 'AttractionId', 'Rating']
df = df.dropna(subset=critical_cols)

# Fill missing categorical values with "Unknown"
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"\nShape after handling missing values: {df.shape}")

# ===================== 3. REMOVE DUPLICATES =====================
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# ===================== 4. CLEAN COLUMN NAMES (Remove suffixes if needed) =====================
# Optional: Rename columns for clarity
df = df.rename(columns={
    'VisitMode_x': 'VisitMode',
    'ContinentId_x': 'ContinentId'
})

# ===================== 5. SAVE CLEANED DATA =====================
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"\n✅ Cleaned data saved to: {CLEANED_DATA_PATH}")
print(f"Final Shape: {df.shape}")

print("\n" + "="*60)
print("DATA CLEANING COMPLETED")
print("="*60)