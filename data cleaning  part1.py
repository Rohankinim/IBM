# to add one hot encoding and target label to binary income 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load your cleaned dataset
df = pd.read_csv("no_dupe_categor.csv")

# Convert PINCP into binary income label: 1 if above median, else 0
median_income = df['PINCP'].median()
df['income_binary'] = (df['PINCP'] > median_income).astype(int)

# One-hot encode RAC1P only
race_ohe = pd.get_dummies(df['RAC1P'], prefix='RACE')

# Drop the original RAC1P
df = df.drop(columns=['RAC1P'])

# Combine everything into a final dataset
df_encoded = pd.concat([df, race_ohe], axis=1)

# Save if needed (optional)
df_encoded.to_csv("acs_encoded_race.csv", index=False)

# Now separate features and target
X = df_encoded.drop(columns=['PINCP', 'income_binary'])  # drop PINCP to prevent leakage
y = df_encoded['income_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
