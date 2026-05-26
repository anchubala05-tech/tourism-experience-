import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / "data" / "processed" / "final_model_data.csv")

# Only use these 6 columns
features = ['VisitMonth', 'Continent', 'Country', 'CityName', 'Attraction', 'AttractionType']
X = df[features]
y = df['VisitMode']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

encoder = TargetEncoder(cols=features)
X_encoded = encoder.fit_transform(X, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save clean files
models_dir = BASE_DIR / "models"
joblib.dump(model, models_dir / "final_clean_model.joblib")
joblib.dump(encoder, models_dir / "final_clean_encoder.joblib")
joblib.dump(le, models_dir / "final_clean_label_encoder.joblib")

print("✅ Clean model trained successfully with 6 features!")