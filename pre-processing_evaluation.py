import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import load
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['PYTHONWARNINGS'] = 'ignore'

def main():
    # 1. Data Loading
    print("📊 Loading data and model for fairness evaluation...")
    try:
        # Load test data (saved during training)
        X_test = pd.read_csv("X_test_fair.csv")
        y_test = pd.read_csv("y_test_fair.csv").squeeze()  # Convert to Series

        # 2. Load fairness-adjusted model
        model = load('fair_logreg_model.joblib')
        
        # 3. Generate predictions
        y_pred = model.predict(X_test)
        
        # 4. Fairness Evaluation
        # Define privileged (White) and unprivileged (non-White) groups
        test_data = StandardDataset(
            df=X_test.assign(income_binary=y_test.values,
                           white_vs_others=(X_test['RACE_1.0'] == 1).astype(int)),
            label_name='income_binary',
            favorable_classes=[1],          # 1 = High Income
            protected_attribute_names=['white_vs_others'],
            privileged_classes=[[1]]        # 1 = White (privileged)
        )
        
        pred_data = test_data.copy()
        pred_data.labels = y_pred.reshape(-1, 1)
        
        metric = ClassificationMetric(
            test_data, pred_data,
            unprivileged_groups=[{'white_vs_others': 0}],  # Non-White
            privileged_groups=[{'white_vs_others': 1}]      # White
        )
        
        # 5. Display Results
        print("\n=== Fairness Metrics ===")
        print(f"{'DIR (Disparate Impact)':<30} {metric.disparate_impact():>10.4f}")
        print(f"{'DPD (Statistical Parity Diff)':<30} {metric.statistical_parity_difference():>10.4f}")
        print(f"{'Equal Opportunity Diff':<30} {metric.equal_opportunity_difference():>10.4f}")
        print(f"{'Average Odds Diff':<30} {metric.average_odds_difference():>10.4f}")
        
        print("\n=== Performance Metrics ===")
        print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred):.4f}")
        print(f"{'F1-Score':<20} {f1_score(y_test, y_pred):.4f}")
        
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, digits=4, 
                                   target_names=['Low Income', 'High Income']))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return

if __name__ == "__main__":
    main()


#  Loading data and model for fairness evaluation...

# === Fairness Metrics ===
# DIR (Disparate Impact)             0.7968
# DPD (Statistical Parity Diff)     -0.1027
# Equal Opportunity Diff            -0.0046
# Average Odds Diff                 -0.0058

# === Performance Metrics ===
# Accuracy             0.9439
# F1-Score             0.9416

# === Classification Report ===
#               precision    recall  f1-score   support

#   Low Income     0.9520    0.9403    0.9461    171205
#  High Income     0.9354    0.9479    0.9416    156033

#     accuracy                         0.9439    327238
#    macro avg     0.9437    0.9441    0.9439    327238
# weighted avg     0.9440    0.9439    0.9440    327238
