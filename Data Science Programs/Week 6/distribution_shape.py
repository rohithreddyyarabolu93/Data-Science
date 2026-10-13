import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("--- Shape of Distribution ---")
print("First 5 rows of dataset:")
print(df.head())

# Skewness (asymmetry of distribution)
# positive skew = long right tail, negative skew = long left tail
skewness_values = df.skew()
print("\nSkewness values:")
print(skewness_values)

# Kurtosis (peakedness / tail heaviness)
# > 0: Leptokurtic (heavy-tailed), < 0: Platykurtic (light-tailed), 0: Mesokurtic (normal)
kurtosis_values = df.kurt()
print("\nKurtosis values:")
print(kurtosis_values)
