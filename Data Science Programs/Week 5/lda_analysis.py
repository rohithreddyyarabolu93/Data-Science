import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data # features
y = iris.target # target labels

# Step 1: Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2) # reduce to 2 dimensions
X_lda = lda.fit_transform(X, y)

# Step 2: Create DataFrame with LDA results
lda_df = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
lda_df['target'] = y

print("Explained variance ratio:", lda.explained_variance_ratio_)
print("\nLDA Result (first 5 rows):")
print(lda_df.head())

# Step 3: Visualize LDA projection
plt.figure(figsize=(8, 6))
for label, color in zip([0, 1, 2], ['red', 'green', 'blue']):
    plt.scatter(
        lda_df.loc[lda_df['target'] == label, 'LD1'],
        lda_df.loc[lda_df['target'] == label, 'LD2'],
        label=iris.target_names[label],
        color=color,
        alpha=0.7
    )

plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('LDA Projection of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()
