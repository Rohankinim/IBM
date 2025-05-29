import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from aif360.metrics import ClassificationMetric

# Constants
SEED = 42
MODEL_PATH = 'C:/IBM/fairness_aware_model/model.pth'
TEST_DATA_PATH = 'C:/IBM/fairness_aware_model/test_data.pkl'
CSV_PATH = 'acs_encoded_race_cleaned.csv'

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Fair model definition
class FairModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.layers(x)

def load_test_data():
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

def load_features_and_protected():
    df = pd.read_csv(CSV_PATH)
    df['is_white'] = np.where(df['RACE_1.0'] == 1, 1, 0)
    
    features = df.drop(columns=['PINCP', 'income_binary', 'is_white'] + [f'RACE_{i}.0' for i in range(1, 10)])
    labels = df['income_binary']
    protected = df['is_white']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return train_test_split_ordered(features_scaled, labels, protected)

def train_test_split_ordered(features, labels, protected):
    from sklearn.model_selection import train_test_split
    return train_test_split(
        features, labels, protected,
        test_size=0.2, random_state=SEED, stratify=labels
    )

def predict(model, X_test):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()
    return y_pred

def evaluate(true_data, pred_labels):
    pred_data = true_data.copy()
    pred_data.labels = pred_labels.reshape(-1, 1)

    metric = ClassificationMetric(
        true_data, pred_data,
        unprivileged_groups=[{'is_white': 0}],
        privileged_groups=[{'is_white': 1}]
    )

    print("\n=== INFERENCE-EVALUATION RESULTS ===")
    print(f"Accuracy: {accuracy_score(true_data.labels, pred_data.labels):.4f}")

    print("\nFAIRNESS METRICS:")
    print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
    print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")
    print(f"Disparate Impact: {metric.disparate_impact():.4f}")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        true_data.labels,
        pred_data.labels,
        target_names=['Low Income', 'High Income'],
        digits=4
    ))

def main():
    print("🔄 Loading model...")
    test_data = load_test_data()
    X_train, X_test, y_train, y_test, prot_train, prot_test = load_features_and_protected()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FairModel(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Model loaded.")

    print("🔍 Making predictions...")
    pred_labels = predict(model, X_test)

    print("📊 Evaluating fairness and accuracy...")
    evaluate(test_data, pred_labels)

if __name__ == "__main__":
    main()



#  python in_processing_part7_evaluation.py
# C:\IBM\myenv\Lib\site-packages\inFairness\utils\ndcg.py:37: FutureWarning: We've integrated functorch into PyTorch. As the final step of the integration, `functorch.vmap` is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use `torch.vmap` instead; see the PyTorch 2.0 release notes and/or the `torch.func` migration guide for more details https://pytorch.org/docs/main/func.migrating.html
#   vect_normalized_discounted_cumulative_gain = vmap(
# C:\IBM\myenv\Lib\site-packages\inFairness\utils\ndcg.py:48: FutureWarning: We've integrated functorch into PyTorch. As the final step of the integration, `functorch.vmap` is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use `torch.vmap` instead; see the PyTorch 2.0 release notes and/or the `torch.func` migration guide for more details https://pytorch.org/docs/main/func.migrating.html
#   monte_carlo_vect_ndcg = vmap(vect_normalized_discounted_cumulative_gain, in_dims=(0,))
# 🔄 Loading model...
# ✅ Model loaded.
# 🔍 Making predictions...
# 📊 Evaluating fairness and accuracy...

# === INFERENCE-EVALUATION RESULTS ===
# Accuracy: 0.7898

# FAIRNESS METRICS:
# Statistical Parity Difference: -0.0452
# Equal Opportunity Difference: 0.0014
# Average Odds Difference: 0.0159
# Disparate Impact: 0.9142

# CLASSIFICATION REPORT:
#               precision    recall  f1-score   support

#   Low Income     0.8239    0.7607    0.7910    171205
#  High Income     0.7578    0.8217    0.7884    156033

#     accuracy                         0.7898    327238
#    macro avg     0.7909    0.7912    0.7897    327238
# weighted avg     0.7924    0.7898    0.7898    327238
