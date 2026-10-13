import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Generating Pair Plot of Iris Features...")

# Pair Plot (relationships among all features)
sns.pairplot(df.iloc[:, :4])
plt.suptitle("Pair Plot of Iris Features", y=1.02)
plt.show()
