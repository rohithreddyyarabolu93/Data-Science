import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Generating Line Chart of Sepal Length across samples...")

# Line Chart (trend of sepal length across samples)
plt.figure(figsize=(6, 4))
plt.plot(df['sepal length (cm)'], color='royalblue', linewidth=1.5)
plt.title("Line Chart of Sepal Length")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
