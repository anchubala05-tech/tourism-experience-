"""
EDA + Data Cleaning for Tourism Project
All graphs will be saved in reports/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===================== SETUP =====================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "master_data.csv"
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(exist_ok=True)   # Create reports folder if not exists

print("Loading master data...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ===================== 1. BASIC INFO =====================
print("\n" + "="*60)
print("1. BASIC INFORMATION")
print("="*60)
print(df.info())

# ===================== 2. MISSING VALUES =====================
print("\n" + "="*60)
print("2. MISSING VALUES CHECK")
print("="*60)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing,
    'Percentage': missing_percent.round(2)
})
print(missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False))

# ===================== 3. RATING DISTRIBUTION =====================
if 'Rating' in df.columns:
    print("\n" + "="*60)
    print("3. RATING DISTRIBUTION")
    print("="*60)
    print(df['Rating'].value_counts().sort_index())

    plt.figure()
    sns.countplot(x='Rating', data=df, palette='viridis')
    plt.title("Distribution of User Ratings")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "rating_distribution.png", dpi=300)
    plt.close()
    print("✅ Saved: reports/rating_distribution.png")

# ===================== 4. VISIT MODE DISTRIBUTION =====================
visit_col = [col for col in df.columns if 'VisitMode' in col and 'Id' not in col]
if visit_col:
    print("\n" + "="*60)
    print("4. VISIT MODE DISTRIBUTION")
    print("="*60)
    print(df[visit_col[0]].value_counts())

    plt.figure()
    df[visit_col[0]].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Visit Mode Distribution")
    plt.xlabel("Visit Mode")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "visit_mode_distribution.png", dpi=300)
    plt.close()
    print("✅ Saved: reports/visit_mode_distribution.png")

# ===================== 5. CORRELATION HEATMAP =====================
print("\n" + "="*60)
print("5. CORRELATION HEATMAP")
print("="*60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "correlation_heatmap.png", dpi=300)
    plt.close()
    print("✅ Saved: reports/correlation_heatmap.png")

# ===================== 6. TOP ATTRACTIONS =====================
if 'Attraction' in df.columns:
    print("\n" + "="*60)
    print("6. TOP 10 ATTRACTIONS BY AVERAGE RATING")
    print("="*60)
    top_attractions = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
    print(top_attractions)

    plt.figure()
    top_attractions.plot(kind='barh', color='teal')
    plt.title("Top 10 Attractions by Average Rating")
    plt.xlabel("Average Rating")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "top_attractions.png", dpi=300)
    plt.close()
    print("✅ Saved: reports/top_attractions.png")

print("\n" + "="*60)
print("✅ EDA Completed! All graphs saved in 'reports/' folder")
print("="*60)