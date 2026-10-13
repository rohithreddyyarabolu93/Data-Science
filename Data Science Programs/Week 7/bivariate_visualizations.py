"""
Week 7: Bivariate Visualizations
================================
1. Scatterplot -> Reveal relationships or correlations between two variables.
2. Line Chart  -> Track trends over time / across samples.
3. Bar Chart   -> Compare categories side by side.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Running Bivariate Visualizations...")

# 1. Scatterplot
print("1. Displaying Scatterplot...")
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], hue=df['species'], palette='viridis')
plt.title("Scatterplot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title='Species', labels=iris.target_names)
plt.tight_layout()
plt.show()

# 2. Line Chart (trend of sepal length across samples)
print("2. Displaying Line Chart...")
plt.figure(figsize=(6, 4))
plt.plot(df['sepal length (cm)'], color='royalblue', linewidth=1.5)
plt.title("Line Chart of Sepal Length")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 3. Bar Chart (mean sepal length per species)
print("3. Displaying Bar Chart...")
plt.figure(figsize=(6, 4))
sns.barplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Set2', legend=False)
plt.title("Bar Chart: Mean Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.tight_layout()
plt.show()

print("Bivariate visualizations completed successfully.")
