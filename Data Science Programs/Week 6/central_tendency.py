import pandas as pd
from sklearn.datasets import load_iris
from statistics import mode

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("--- Measures of Central Tendency ---")
print("First 5 rows of dataset:")
print(df.head())

# Mean (average)
mean_values = df.mean()
print("\nMean values:")
print(mean_values)

# Median (middle value)
median_values = df.median()
print("\nMedian values:")
print(median_values)

# Mode (most frequent value)
# Pandas mode returns a DataFrame (can be multiple modes)
mode_values = df.mode().iloc[0]
print("\nMode values:")
print(mode_values)
