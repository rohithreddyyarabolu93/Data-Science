import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data # features
y = iris.target # target labels

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    max_iter=1000
)
X_tsne = tsne.fit_transform(X)

# Create DataFrame with t-SNE results
tsne_df = pd.DataFrame(
    X_tsne,
    columns=['TSNE1', 'TSNE2']
)
tsne_df['target'] = y

# Display first 5 rows
print("t-SNE Result (first 5 rows):")
print(tsne_df.head())

# Visualize t-SNE projection
plt.figure(figsize=(8, 6))
for label, color in zip(
    [0, 1, 2],
    ['red', 'green', 'blue']
):
    plt.scatter(
        tsne_df.loc[tsne_df['target'] == label, 'TSNE1'],
        tsne_df.loc[tsne_df['target'] == label, 'TSNE2'],
        label=iris.target_names[label],
        color=color,
        alpha=0.7
    )

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Projection of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()
