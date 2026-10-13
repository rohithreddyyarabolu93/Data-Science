import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Generating Violin Plot: Sepal Length by Species...")

# Violin Plot
plt.figure(figsize=(6, 4))
sns.violinplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Pastel1', legend=False)
plt.title("Violin Plot: Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.tight_layout()
plt.show()
