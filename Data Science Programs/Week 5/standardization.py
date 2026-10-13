import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example dataset
data = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [5, 15, 25, 35, 45]
})

# Apply Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data) # fit and transform the data

# Convert back to DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=data.columns)

print("\nStandardized Data (Z-score):")
print(standardized_df)
