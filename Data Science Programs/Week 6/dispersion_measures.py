import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("--- Measures of Dispersion ---")
print("First 5 rows of dataset:")
print(df.head())

# Range (max - min)
range_values = df.max() - df.min()
print("\nRange values:")
print(range_values)

# Variance
variance_values = df.var()
print("\nVariance values:")
print(variance_values)

# Standard Deviation
std_values = df.std()
print("\nStandard Deviation values:")
print(std_values)

# Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR):")
print(IQR)
