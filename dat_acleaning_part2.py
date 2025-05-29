# data cleaning and validation

import pandas as pd

# Load encoded data
df = pd.read_csv("acs_encoded_race.csv")

# ----- 1. Check for missing values -----
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Optional: Drop or impute missing values if any (none expected if pre-cleaned)
# df = df.dropna()

# ----- 2. Check for duplicate rows -----
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Optional: Drop duplicates
df = df.drop_duplicates()

# ----- 3. Check data types -----
print("Data types:\n", df.dtypes)

# ----- 4. Check unique values in categorical columns (for encoded race columns) -----
race_columns = [col for col in df.columns if col.startswith("RACE_")]
print("Race encoding columns:\n", df[race_columns].sum())

# Confirm that each row has only one race marked as 1
race_sum = df[race_columns].sum(axis=1)
invalid_race_rows = df[race_sum != 1]
print(f"Rows with invalid one-hot encoding for race: {len(invalid_race_rows)}")

# ----- 5. Check income_binary target -----
print(df['income_binary'].value_counts())

# Optional: Check for severe imbalance
imbalance_ratio = df['income_binary'].value_counts(normalize=True)
print("Income class imbalance ratio:\n", imbalance_ratio)

# ----- 6. Check for outliers in numeric columns (Optional) -----
# If you have numeric columns like age, income etc.
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Summary statistics:\n", df[numeric_cols].describe())

import pandas as pd

# or pd.read_excel(), etc.

# Get shape (rows, columns)
print("Dataset shape:", df.shape)

# Get size (total number of elements)
print("Dataset size:", df.size)

# Get dimensions
print("Number of dimensions:", df.ndim)

# Display info including memory usage
df.info()

df.to_csv("acs_encoded_race_cleaned.csv", index=False)
print("Cleaned dataset saved.")
