import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

# Load dataset
tips = sns.load_dataset("tips") # or pd.read_csv("tips.csv")

print("Original Data (first 5 rows):")
print(tips.head())

# 1. Normalization (Min-Max Scaling)
numeric_cols = tips.select_dtypes(include=['float64', 'int64']).columns # Selects only numeric columns ('total_bill', 'tip', 'size')

scaler_minmax = MinMaxScaler() # Create MinMaxScaler Object
tips_normalized = tips.copy() # A copy is created to preserve the original data
tips_normalized[numeric_cols] = scaler_minmax.fit_transform(tips[numeric_cols])

print("\nNormalized Data (first 5 rows):")
print(tips_normalized.head())

# 2. Standardization (Z-score)
scaler_standard = StandardScaler()
tips_standardized = tips.copy()
tips_standardized[numeric_cols] = scaler_standard.fit_transform(tips[numeric_cols])

print("\nStandardized Data (first 5 rows):")
print(tips_standardized.head())

# 3. Encoding Categorical Variables
# (a) One-Hot Encoding
tips_onehot = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])

print("\nOne-Hot Encoded Data (first 5 rows):")
print(tips_onehot.head())
