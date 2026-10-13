"""
Week 7: Multivariate Visualizations
===================================
1. Heatmaps     -> Display correlation matrices or intensity values.
2. Bubble Charts-> Add a third variable (size) to scatterplots.
3. Pair Plots   -> Show multiple variable relationships simultaneously.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Running Multivariate Visualizations...")

# 1. Heatmap (correlation matrix)
print("1. Displaying Heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.tight_layout()
plt.show()

# 2. Bubble Chart (scatterplot with size = petal length)
print("2. Displaying Bubble Chart...")
plt.figure(figsize=(6, 4))
plt.scatter(
    df['sepal length (cm)'], 
    df['sepal width (cm)'], 
    s=df['petal length (cm)'] * 20, 
    alpha=0.5, 
    c=df['species'], 
    cmap='viridis'
)
plt.title("Bubble Chart: Sepal vs Petal (size = petal length)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# 3. Pair Plot (relationships among all features)
print("3. Displaying Pair Plot...")
sns.pairplot(df.iloc[:, :4])
plt.suptitle("Pair Plot of Iris Features", y=1.02)
plt.show()

print("Multivariate visualizations completed successfully.")
