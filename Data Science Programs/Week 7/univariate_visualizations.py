"""
Week 7: Univariate Visualizations
=================================
1. Histogram  -> Show distribution of a single variable.
2. Boxplot    -> Highlight spread, median, and outliers.
3. Pie Chart  -> Represent proportions of categorical data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Running Univariate Visualizations...")

# 1. Histogram
print("1. Displaying Histogram...")
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, color='teal')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count / Density")
plt.tight_layout()
plt.show()

# 2. Boxplot
print("2. Displaying Boxplot...")
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['sepal length (cm)'], color='lightblue')
plt.title("Boxplot of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Pie Chart (species distribution)
print("3. Displaying Pie Chart...")
species_counts = df['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=iris.target_names, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title("Species Distribution (Pie Chart)")
plt.tight_layout()
plt.show()

print("Univariate visualizations completed successfully.")
