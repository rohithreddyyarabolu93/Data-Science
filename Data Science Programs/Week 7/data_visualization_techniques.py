"""
Week 7: Implement & Analyze Data Visualization Techniques for Effective Interpretation of Datasets
==================================================================================================
This master script implements all the core data visualization techniques:

1. Univariate Visualizations:
   - Histograms   -> Show distribution of a single variable.
   - Boxplots     -> Highlight spread, median, and outliers.
   - Pie Charts   -> Represent proportions of categorical data.

2. Bivariate Visualizations:
   - Scatterplots -> Reveal relationships or correlations between two variables.
   - Line Charts  -> Track trends over time / across samples.
   - Bar Charts   -> Compare categories side by side.

3. Multivariate Visualizations:
   - Heatmaps     -> Display correlation matrices or intensity values.
   - Bubble Charts-> Add a third variable (size) to scatterplots.
   - Pair Plots   -> Show multiple variable relationships simultaneously.

4. Advanced Techniques:
   - Violin Plots -> Combine boxplot and density distribution.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ==========================================
# Load Iris Dataset
# ==========================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 75)
print("WEEK 7: DATA VISUALIZATION TECHNIQUES FOR EFFECTIVE INTERPRETATION")
print("=" * 75)
print("\nFirst 5 rows of Iris dataset:")
print(df.head())
print("\nDataset Summary Statistics:")
print(df.describe())

# ==========================================
# 1. Univariate Visualizations
# ==========================================
print("\n" + "-" * 40)
print("1. UNIVARIATE VISUALIZATIONS")
print("-" * 40)

# Histogram
print("Plotting Histogram of Sepal Length...")
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, color='teal')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot
print("Plotting Boxplot of Sepal Length...")
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['sepal length (cm)'], color='lightblue')
plt.title("Boxplot of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.tight_layout()
plt.show()

# Pie Chart (species distribution)
print("Plotting Pie Chart of Species Distribution...")
species_counts = df['species'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(species_counts, labels=iris.target_names, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title("Species Distribution (Pie Chart)")
plt.tight_layout()
plt.show()

# ==========================================
# 2. Bivariate Visualizations
# ==========================================
print("\n" + "-" * 40)
print("2. BIVARIATE VISUALIZATIONS")
print("-" * 40)

# Scatterplot
print("Plotting Scatterplot: Sepal Length vs Sepal Width...")
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], hue=df['species'], palette='viridis')
plt.title("Scatterplot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title='Species', labels=iris.target_names)
plt.tight_layout()
plt.show()

# Line Chart (trend of sepal length across samples)
print("Plotting Line Chart of Sepal Length...")
plt.figure(figsize=(6, 4))
plt.plot(df['sepal length (cm)'], color='royalblue', linewidth=1.5)
plt.title("Line Chart of Sepal Length")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Bar Chart (mean sepal length per species)
print("Plotting Bar Chart: Mean Sepal Length by Species...")
plt.figure(figsize=(6, 4))
sns.barplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Set2', legend=False)
plt.title("Bar Chart: Mean Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.tight_layout()
plt.show()

# ==========================================
# 3. Multivariate Visualizations
# ==========================================
print("\n" + "-" * 40)
print("3. MULTIVARIATE VISUALIZATIONS")
print("-" * 40)

# Heatmap (correlation matrix)
print("Plotting Heatmap of Feature Correlations...")
plt.figure(figsize=(8, 6))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Feature Correlations")
plt.tight_layout()
plt.show()

# Bubble Chart (scatterplot with size = petal length)
print("Plotting Bubble Chart: Sepal vs Petal...")
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

# Pair Plot (relationships among all features)
print("Plotting Pair Plot of Iris Features...")
sns.pairplot(df.iloc[:, :4])
plt.suptitle("Pair Plot of Iris Features", y=1.02)
plt.show()

# ==========================================
# 4. Advanced Techniques
# ==========================================
print("\n" + "-" * 40)
print("4. ADVANCED TECHNIQUES")
print("-" * 40)

# Violin Plot
print("Plotting Violin Plot: Sepal Length by Species...")
plt.figure(figsize=(6, 4))
sns.violinplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Pastel1', legend=False)
plt.title("Violin Plot: Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.tight_layout()
plt.show()

print("\n" + "=" * 75)
print("ALL WEEK 7 DATA VISUALIZATION TECHNIQUES SUCCESSFULLY EXECUTED")
print("=" * 75)
