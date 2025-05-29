import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import warnings
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data():
    """Load and preprocess data with error handling"""
    try:
        df = pd.read_csv("acs_encoded_race_cleaned.csv", engine='c')
        required = {'PINCP', 'income_binary'}.union({f'RACE_{i}.0' for i in range(1,10)})
        
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        
        df['is_white'] = np.where(df['RACE_1.0'] == 1, 1, 0)
        features = df.drop(columns=['PINCP', 'income_binary', 'is_white'] + [f'RACE_{i}.0' for i in range(1,10)])
        return features, df['income_binary'], df['is_white']
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def create_dataset(features, labels, protected):
    """Create AIF360 StandardDataset with error handling"""
    try:
        df = features.copy()
        df['income'] = labels.values
        df['is_white'] = protected.values
        return StandardDataset(
            df,
            label_name='income',
            favorable_classes=[1],
            protected_attribute_names=['is_white'],
            privileged_classes=[[1]]
        )
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        raise

class FairModel(nn.Module):
    """Optimized fair model with better architecture"""
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

def fairness_aware_loss(y_pred, y_true, group_labels, lambda_fair=0.5):
    """Optimized fairness-aware loss"""
    # Standard classification loss
    base_loss = nn.CrossEntropyLoss()(y_pred, y_true)
    
    # Fairness penalty
    probs = torch.softmax(y_pred, dim=1)[:, 1]
    group_0 = probs[group_labels == 0]
    group_1 = probs[group_labels == 1]
    
    # Handle empty groups
    rate_0 = group_0.mean() if len(group_0) > 0 else torch.tensor(0.0)
    rate_1 = group_1.mean() if len(group_1) > 0 else torch.tensor(0.0)
    
    fairness_penalty = torch.abs(rate_0 - rate_1)
    return base_loss + lambda_fair * fairness_penalty

def train_and_save_model(X_train, y_train, prot_train, num_epochs=150, batch_size=2048, lambda_fair=0.5):
    """Optimized training function"""
    try:
        # Convert to tensors and move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train.values).to(device)
        prot_tensor = torch.LongTensor(prot_train.values).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor, prot_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        model = FairModel(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in tqdm(range(num_epochs), desc="Training"):
            total_loss = 0
            for X_batch, y_batch, prot_batch in dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = fairness_aware_loss(y_pred, y_batch, prot_batch, lambda_fair)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save model
        save_path = 'C:/IBM/fairness_aware_model'
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
        print(f"Model saved to {save_path}/model.pth")
        
        return model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def predict(model, X_test):
    """Optimized prediction function"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_tensor).argmax(dim=1).cpu().numpy()
    return y_pred

def evaluate(true_data, pred_labels, X_test, prot_test):
    """Evaluation function"""
    try:
        pred_data = true_data.copy()
        pred_data.labels = pred_labels.reshape(-1, 1)
        
        metric = ClassificationMetric(
            true_data, pred_data,
            unprivileged_groups=[{'is_white': 0}],
            privileged_groups=[{'is_white': 1}]
        )
        
        print("\n=== FAIRNESS-AWARE RESULTS ===")
        print(f"Accuracy: {accuracy_score(true_data.labels, pred_data.labels):.4f}")
        
        print("\nFAIRNESS METRICS:")
        print(f"Statistical Parity: {metric.statistical_parity_difference():.4f}")
        print(f"Equal Opportunity: {metric.equal_opportunity_difference():.4f}")
        print(f"Average Odds: {metric.average_odds_difference():.4f}")
        print(f"Disparate Impact: {metric.disparate_impact():.4f}")
        
        print("\nCLASSIFICATION PERFORMANCE:")
        print(classification_report(
            true_data.labels,
            pred_data.labels,
            target_names=['Low Income', 'High Income'],
            digits=4
        ))
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def main():
    try:
        print("Loading data...")
        features, labels, protected = load_data()
        
        print("Scaling features...")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            features, labels, protected, test_size=0.2, random_state=SEED, stratify=labels)
        
        print("Creating test dataset...")
        test_data = create_dataset(pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])]), 
                                 y_test, prot_test)
        
        with open("test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
        print("✅ test_data.pkl saved successfully.")
        
        print("\nTraining model...")
        model = train_and_save_model(X_train, y_train, prot_train, 
                                   num_epochs=50, batch_size=4096, lambda_fair=0.5)
        
        print("\nMaking predictions...")
        pred_labels = predict(model, X_test)
        
        print("\nEvaluating results...")
        evaluate(test_data, pred_labels, X_test, prot_test)
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()



# Scaling features...
# Splitting data...
# Creating test dataset...
# ✅ test_data.pkl saved successfully.

# Training model...
# Training:  18%|███████████▉                                                      | 9/50 [04:34<20:47, 30.43s/it]Epoch 10, Loss: 0.4623
# Training:  38%|████████████████████████▋                                        | 19/50 [09:43<16:27, 31.84s/it]Epoch 20, Loss: 0.4537
# Training:  58%|█████████████████████████████████████▋                           | 29/50 [14:53<10:36, 30.30s/it]Epoch 30, Loss: 0.4499
# Training:  78%|██████████████████████████████████████████████████▋              | 39/50 [20:09<06:14, 34.07s/it]Epoch 40, Loss: 0.4482
# Training:  98%|███████████████████████████████████████████████████████████████▋ | 49/50 [26:02<00:34, 34.94s/it]Epoch 50, Loss: 0.4467
# Training: 100%|█████████████████████████████████████████████████████████████████| 50/50 [26:33<00:00, 31.86s/it]
# Model saved to C:/IBM/fairness_aware_model/model.pth

# Making predictions...

# Evaluating results...

# === FAIRNESS-AWARE RESULTS ===
# Accuracy: 0.7898

# FAIRNESS METRICS:
# Statistical Parity: -0.0452
# Equal Opportunity: 0.0014
# Average Odds: 0.0159
# Disparate Impact: 0.9142

# CLASSIFICATION PERFORMANCE:
#               precision    recall  f1-score   support

#   Low Income     0.8239    0.7607    0.7910    171205
#  High Income     0.7578    0.8217    0.7884    156033

#     accuracy                         0.7898    327238
#    macro avg     0.7909    0.7912    0.7897    327238
# weighted avg     0.7924    0.7898    0.7898    327238