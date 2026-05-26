"""
Tourism Experience Analytics - Data Loader (Final Fixed Version)
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

def load_file(filename):
    filepath = RAW_PATH / filename
    if not filepath.exists():
        print(f"❌ File NOT found: {filename}")
        return None
    df = pd.read_excel(filepath)
    print(f"✅ Loaded: {filename} | Rows: {len(df)}")
    return df

def create_master_data():
    print("\n🚀 Starting Data Loading & Merging...\n")

    df_trans     = load_file("Transaction.xlsx")
    df_user      = load_file("User.xlsx")
    df_city      = load_file("City.xlsx")
    df_item      = load_file("Updated_Item.xlsx")
    df_type      = load_file("Type.xlsx")
    df_mode      = load_file("Mode.xlsx")
    df_country   = load_file("Country.xlsx")
    df_continent = load_file("Continent.xlsx")
    df_region    = load_file("Region.xlsx")

    if df_trans is None:
        return

    master = df_trans.copy()

    # Merge in safe order
    if df_user is not None:
        master = master.merge(df_user, on="UserId", how="left", suffixes=("", "_user"))

    if df_city is not None:
        master = master.merge(df_city, on="CityId", how="left", suffixes=("", "_city"))

    if df_item is not None:
        master = master.merge(df_item, on="AttractionId", how="left", suffixes=("", "_item"))

    if df_type is not None:
        master = master.merge(df_type, on="AttractionTypeId", how="left")

    if df_mode is not None:
        master = master.merge(df_mode, left_on="VisitMode", right_on="VisitModeId", how="left")

    # Merge lookup tables (use suffixes to avoid column conflicts)
    if df_country is not None:
        master = master.merge(df_country, on="CountryId", how="left", suffixes=("", "_country"))

    if df_continent is not None:
        master = master.merge(df_continent, on="ContinentId", how="left")

    if df_region is not None:
        master = master.merge(df_region, on="RegionId", how="left")

    master = master.drop_duplicates()

    print(f"\n✅ Master Data Created | Final Rows: {len(master)}")
    print("Sample columns:", list(master.columns)[:15])

    output_path = PROCESSED_PATH / "master_data.csv"
    master.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")

if __name__ == "__main__":
    create_master_data()