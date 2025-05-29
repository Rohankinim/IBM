import joblib
from aif360.metrics import ClassificationMetric
import warnings
import os

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Directory where model is saved
MODEL_DIR = "baseline_no_mitigation"

# Load model and test data
print("🔄 Loading model and test data...")
model = joblib.load(f"{MODEL_DIR}/baseline_model.joblib")
X_test, y_test, aif_test = joblib.load(f"{MODEL_DIR}/test_data.joblib")

# Make predictions
print("\n🔮 Making predictions...")
y_pred = model.predict(X_test)
aif_pred = aif_test.copy()
aif_pred.labels = y_pred.reshape(-1, 1)

# Fairness evaluation
print("\n📊 Evaluating fairness...")
metric = ClassificationMetric(
    aif_test,
    aif_pred,
    unprivileged_groups=[{'white_vs_others': 0}],
    privileged_groups=[{'white_vs_others': 1}]
)

print("\n=== BASELINE FAIRNESS METRICS (No Mitigation) ===")
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(f"Disparate Impact Ratio (DIR): {metric.disparate_impact():.4f}")
print(f"Demographic Parity Difference (DPD): {metric.statistical_parity_difference():.4f}")
print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")




#  Making predictions...
# [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
# [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    4.0s
# [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:   10.8s finished

# 📊 Evaluating fairness...

# === BASELINE FAIRNESS METRICS (No Mitigation) ===
# [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
# [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    3.5s
# [Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    8.9s finished
# Accuracy: 0.8014
# Disparate Impact Ratio (DIR): 0.7047
# Demographic Parity Difference (DPD): -0.1507
# Equal Opportunity Difference: -0.0993
# Average Odds Difference: -0.0874