import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Generating Bubble Chart: Sepal Length vs Sepal Width with Bubble Size = Petal Length...")

# Bubble Chart (scatterplot with size = petal length)
plt.figure(figsize=(6, 4))
scatter = plt.scatter(
    df['sepal length (cm)'], 
    df['sepal width (cm)'], 
    s=df['petal length (cm)'] * 20, 
    alpha=0.6, 
    c=df['species'], 
    cmap='viridis',
    edgecolors='black'
)
plt.title("Bubble Chart: Sepal vs Petal (size = petal length)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
cbar = plt.colorbar(scatter, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(iris.target_names)
cbar.set_label('Species')
plt.tight_layout()
plt.show()
