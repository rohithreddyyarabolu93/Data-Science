"""
Week 7: Data Visualization Summary Dashboard
=============================================
This script produces a consolidated multi-panel dashboard saving all 
Univariate, Bivariate, Multivariate, and Advanced visualizations into an image.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Week 7: Data Visualization Techniques Dashboard", fontsize=18, fontweight='bold', y=0.98)

# 1. Univariate: Histogram
ax1 = plt.subplot(3, 3, 1)
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, color='teal', ax=ax1)
ax1.set_title("1. Histogram (Sepal Length)")
ax1.set_xlabel("Sepal Length (cm)")

# 2. Univariate: Boxplot
ax2 = plt.subplot(3, 3, 2)
sns.boxplot(x=df['sepal length (cm)'], color='skyblue', ax=ax2)
ax2.set_title("2. Boxplot (Sepal Length)")
ax2.set_xlabel("Sepal Length (cm)")

# 3. Univariate: Pie Chart
ax3 = plt.subplot(3, 3, 3)
species_counts = df['species'].value_counts()
ax3.pie(species_counts, labels=iris.target_names, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])
ax3.set_title("3. Pie Chart (Species Proportions)")

# 4. Bivariate: Scatterplot
ax4 = plt.subplot(3, 3, 4)
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species_name', palette='viridis', ax=ax4)
ax4.set_title("4. Scatterplot (Sepal Length vs Width)")
ax4.set_xlabel("Sepal Length (cm)")
ax4.set_ylabel("Sepal Width (cm)")

# 5. Bivariate: Line Chart
ax5 = plt.subplot(3, 3, 5)
ax5.plot(df['sepal length (cm)'], color='royalblue', linewidth=1.2)
ax5.set_title("5. Line Chart (Sepal Length across Samples)")
ax5.set_xlabel("Sample Index")
ax5.set_ylabel("Sepal Length (cm)")
ax5.grid(True, linestyle='--', alpha=0.5)

# 6. Bivariate: Bar Chart
ax6 = plt.subplot(3, 3, 6)
sns.barplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Set2', legend=False, ax=ax6)
ax6.set_title("6. Bar Chart (Mean Sepal Length by Species)")
ax6.set_xticks([0, 1, 2])
ax6.set_xticklabels(iris.target_names)
ax6.set_ylabel("Mean Sepal Length (cm)")

# 7. Multivariate: Heatmap
ax7 = plt.subplot(3, 3, 7)
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax7)
ax7.set_title("7. Heatmap (Correlation Matrix)")

# 8. Multivariate: Bubble Chart
ax8 = plt.subplot(3, 3, 8)
sc = ax8.scatter(
    df['sepal length (cm)'], 
    df['sepal width (cm)'], 
    s=df['petal length (cm)'] * 25, 
    c=df['species'], 
    cmap='plasma', 
    alpha=0.6,
    edgecolors='black',
    linewidths=0.5
)
ax8.set_title("8. Bubble Chart (Size = Petal Length)")
ax8.set_xlabel("Sepal Length (cm)")
ax8.set_ylabel("Sepal Width (cm)")

# 9. Advanced: Violin Plot
ax9 = plt.subplot(3, 3, 9)
sns.violinplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Pastel1', legend=False, ax=ax9)
ax9.set_title("9. Violin Plot (Sepal Length by Species)")
ax9.set_xticks([0, 1, 2])
ax9.set_xticklabels(iris.target_names)
ax9.set_ylabel("Sepal Length (cm)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

output_dir = os.path.dirname(os.path.abspath(__file__))
output_img = os.path.join(output_dir, "visualization_dashboard.png")
plt.savefig(output_img, dpi=300)
print(f"Dashboard saved successfully to {output_img}")
plt.show()
