import pandas as pd

# Example dataset
df = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 24, 33, 45, 60]
})

# Pearson correlation matrix
corr_matrix = df.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)
