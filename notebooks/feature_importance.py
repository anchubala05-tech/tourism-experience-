"""
Feature Importance Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).parent.parent
REPORTS_PATH = BASE_DIR / "reports"
REPORTS_PATH.mkdir(exist_ok=True)

# Load best model and encoder
model = joblib.load(BASE_DIR / "models" / "best_tuned_xgboost.joblib")
encoder = joblib.load(BASE_DIR / "models" / "target_encoder.joblib")

# Get feature names (after encoding)
feature_names = encoder.get_feature_names_out()

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feat_imp)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance - Best Tuned XGBoost Model")
plt.tight_layout()
plt.savefig(REPORTS_PATH / "feature_importance.png", dpi=300)
plt.show()

print("\n✅ Feature importance plot saved in reports/feature_importance.png")