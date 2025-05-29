# train_logreg_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

def main():
    print("🚀 Loading and preparing dataset...")
    df = pd.read_csv("acs_encoded_race_cleaned.csv")

    required_cols = [f'RACE_{i}.0' for i in range(1, 10)] + ['income_binary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    X = df.drop(columns=['income_binary'])
    y = df['income_binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("🧠 Training Logistic Regression model...")
    clf = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        C=0.1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("💾 Saving model and test data...")
    joblib.dump(clf, "logreg_model.joblib")
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

    print("✅ Model and test data saved successfully.")

if __name__ == "__main__":
    main()
