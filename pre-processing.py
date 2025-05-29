import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
import warnings

warnings.filterwarnings("ignore")

def main():
    print("🚀 Loading and preparing dataset...")
    df = pd.read_csv("acs_encoded_race_cleaned.csv")

    # Check required columns (including race and income)
    required_cols = ['RACE_1.0', 'income_binary'] + [f'RACE_{i}.0' for i in range(2, 10)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Split data (stratified by income)
    X = df.drop(columns=['income_binary'])
    y = df['income_binary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Convert to AIF360 StandardDataset for fairness processing
    train_data = StandardDataset(
        df=X_train.assign(income_binary=y_train.values),
        label_name='income_binary',
        favorable_classes=[1],
        protected_attribute_names=['RACE_1.0'],
        privileged_classes=[[1]]  # 1 = White (privileged group)
    )

    # Apply reweighing to balance outcomes across races
    print("⚖️ Applying reweighing for fairness...")
    RW = Reweighing(
        unprivileged_groups=[{"RACE_1.0": 0}],  # Non-White
        privileged_groups=[{"RACE_1.0": 1}]      # White
    )
    train_data_transformed = RW.fit_transform(train_data)

    # Extract reweighted samples
    X_train_fair = train_data_transformed.features
    y_train_fair = train_data_transformed.labels.ravel()
    sample_weights = train_data_transformed.instance_weights

    # Train Logistic Regression with fairness-aware weights
    print("🧠 Training fairness-adjusted Logistic Regression...")
    clf = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        class_weight=None,  # Disable class_weight (handled by reweighing)
        C=0.1,
        random_state=42
    )
    clf.fit(X_train_fair, y_train_fair, sample_weight=sample_weights)

    # Save model and test data
    print("💾 Saving model and test data...")
    joblib.dump(clf, "fair_logreg_model.joblib")
    X_test.to_csv("X_test_fair.csv", index=False)
    y_test.to_csv("y_test_fair.csv", index=False)

    print("✅ Fairness-adjusted model saved successfully!")

if __name__ == "__main__":
    main()