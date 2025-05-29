import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv('data/processed/acs_2018_cleaned.csv')

# Define labels with corrected quotes
labels = {
    'SEX': {1: 'Male', 2: 'Female'},
    'RAC1P': {1: 'White', 2: 'Black', 3: 'American Indian', 4: 'Alaska Native', 5: 'Indian/Alaska Native', 6: 'Asian', 7: 'Pacific Islander', 8: 'Other', 9: 'Two+ races'},
    'SCHL': {1: 'No schooling', 2: 'Nursery', 3: 'Kindergarten', 4: 'Grade 1', 5: 'Grade 2', 6: 'Grade 3', 7: 'Grade 4', 8: 'Grade 5', 9: 'Grade 6', 10: 'Grade 7', 11: 'Grade 8', 12: 'Grade 9', 13: 'Grade 10', 14: 'Grade 11', 15: 'Grade 12', 16: 'High school', 17: 'GED', 18: 'Some college <1yr', 19: 'College no degree', 20: 'Associate\'s', 21: 'Bachelor\'s', 22: 'Master\'s', 23: 'Professional', 24: 'Doctorate'},
    'MAR': {1: 'Married', 2: 'Widowed', 3: 'Divorced', 4: 'Separated', 5: 'Never married/under 15'}
}

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# Optional: Sample data for faster plotting (uncomment if needed)
# df = df.sample(frac=0.1, random_state=42)  # 10% sample

# PINCP Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['PINCP'], bins=50)
plt.title('Distribution of Personal Income (PINCP)')
plt.xlabel('Income ($)')
plt.ylabel('Count')
plt.savefig('visualizations/pincp_distribution.png')
plt.close()

# Count Plots
for col in ['SEX', 'RAC1P', 'SCHL', 'MAR']:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[col].map(labels[col]), data=df)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'visualizations/{col}_distribution.png')
    plt.close()

# Boxplots of PINCP by Category
for col in ['SEX', 'RAC1P', 'SCHL', 'MAR']:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df[col].map(labels[col]), y='PINCP', data=df)
    plt.title(f'PINCP by {col}')
    plt.xlabel(col)
    plt.ylabel('Income ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'visualizations/pincp_by_{col}.png')
    plt.close()

# Mean PINCP Bar Plots
for col in ['SEX', 'RAC1P', 'SCHL', 'MAR']:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df[col].map(labels[col]), y='PINCP', data=df, errorbar=None)
    plt.title(f'Mean PINCP by {col}')
    plt.xlabel(col)
    plt.ylabel('Mean Income ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'visualizations/mean_pincp_by_{col}.png')
    plt.close()

# Summarize mean PINCP by group
print("EDA Summary:")
for col in ['SEX', 'RAC1P', 'SCHL', 'MAR']:
    print(f"\nMean PINCP by {col}:")
    print(df.groupby(df[col].map(labels[col]))['PINCP'].mean().round(2))