import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Generating Bar Chart: Mean Sepal Length by Species...")

# Bar Chart (mean sepal length per species)
plt.figure(figsize=(6, 4))
sns.barplot(x=df['species'], y=df['sepal length (cm)'], hue=df['species'], palette='Set2', legend=False)
plt.title("Bar Chart: Mean Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.xticks([0, 1, 2], iris.target_names)
plt.tight_layout()
plt.show()
