# evaluate_logreg_model.py
import pandas as pd
import numpy as np
import joblib
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import warnings

warnings.filterwarnings("ignore")

def create_aif_dataset(X, y):
    df_aif = X.copy()
    df_aif['income'] = y.values
    df_aif['race'] = np.where(df_aif['RACE_1.0'] == 1, 1, 0)
    return StandardDataset(
        df=df_aif,
        label_name='income',
        favorable_classes=[1],
        protected_attribute_names=['race'],
        privileged_classes=[[1]]
    )

def print_metrics(metric):
    print("\n=== Evaluation After Post-processing ===")
    print(f"Accuracy: {metric.accuracy():.4f}")
    print("Fairness Metrics:")
    print(f"DIR: {metric.disparate_impact():.4f}")
    print(f"DPD: {metric.statistical_parity_difference():.4f}")
    print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")

def main():
    print("📦 Loading model and test data...")
    clf = joblib.load("logreg_model.joblib")
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv").squeeze()

    print("🎯 Generating predictions...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("📊 Creating AIF360 datasets...")
    aif_test = create_aif_dataset(X_test, y_test)
    aif_pred = aif_test.copy()
    aif_pred.labels = y_pred.reshape(-1, 1)
    aif_pred.scores = y_prob.reshape(-1, 1)

    print("\n⚖️ Applying Calibrated Equalized Odds Postprocessing...")
    cpp = CalibratedEqOddsPostprocessing(
        privileged_groups=[{'race': 1}],
        unprivileged_groups=[{'race': 0}],
        cost_constraint='weighted',
        seed=42
    )
    cpp.fit(aif_test, aif_pred)
    aif_pred_eq = cpp.predict(aif_pred)

    metric_post = ClassificationMetric(
        aif_test, aif_pred_eq,
        unprivileged_groups=[{'race': 0}],
        privileged_groups=[{'race': 1}]
    )
    print_metrics(metric_post)

if __name__ == "__main__":
    main()


# === Evaluation After Post-processing ===
# Accuracy: 0.9410
# Fairness Metrics:
# DIR: 0.7960
# DPD: -0.1043
# Equal Opportunity Difference: -0.0073
# Average Odds Difference: -0.0080