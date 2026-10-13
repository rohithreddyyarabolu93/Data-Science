import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Generating Scatterplot: Sepal Length vs Sepal Width...")

# Scatterplot
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], hue=df['species'], palette='viridis')
plt.title("Scatterplot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title='Species', labels=iris.target_names)
plt.tight_layout()
plt.show()
