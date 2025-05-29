import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from joblib import parallel_backend
import os
import warnings
import joblib

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Create directory for saving models
MODEL_DIR = "baseline_no_mitigation"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print("🔄 Loading data...")
df = pd.read_csv("acs_encoded_race_cleaned.csv")

# Check available columns
print("\n📋 Available columns in dataset:")
print(df.columns.tolist())

# Verify RACE_1.0 column exists (White group)
if 'RACE_1.0' not in df.columns:
    raise ValueError("❌ 'RACE_1.0' column not found. Required for White vs others comparison.")

# Separate features and target
X = df.drop(columns=['PINCP', 'income_binary'])
y = df['income_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.3,
    random_state=42
)

# Convert to AIF360 dataset format
def to_aif360_dataset(X, y):
    df_aif = X.copy()
    df_aif['income'] = y.values
    df_aif['white_vs_others'] = np.where(df_aif['RACE_1.0'] == 1, 1, 0)

    return StandardDataset(
        df_aif,
        label_name='income',
        favorable_classes=[1],
        protected_attribute_names=['white_vs_others'],
        privileged_classes=[[1]]
    )

# Create AIF360 datasets
print("\n🔧 Creating AIF360 dataset...")
aif_train = to_aif360_dataset(X_train, y_train)
aif_test = to_aif360_dataset(X_test, y_test)

# Save test data for evaluation
joblib.dump((X_test, y_test, aif_test), f"{MODEL_DIR}/test_data.joblib")

# Train baseline model
print("\n🏗️  Training baseline model...")
with parallel_backend('threading', n_jobs=-1):
    model = RandomForestClassifier(
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    model.fit(X_train, y_train)

# Save the model
model_path = f"{MODEL_DIR}/baseline_model.joblib"
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}")