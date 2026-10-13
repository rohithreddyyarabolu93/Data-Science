"""
Week 3: Implementation of Similarity and Dissimilarity Measures for Data Analysis
Executes all codes directly from the PDF.
"""

# ----------------------------------------------------
# 3.1 Vector-Based Distance Measures
# ----------------------------------------------------
print("=" * 60)
print("3.1.1 Euclidean Distance & Similarity")
print("=" * 60)
import numpy as np
from scipy.spatial import distance

pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Euclidean Distance
euclidean_dist = distance.euclidean(pointA, pointB)
print("Euclidean Distance:", euclidean_dist)

# Similarity (inverse of distance)
similarity_euclidean = 1 / (1 + euclidean_dist)
print("Euclidean Similarity:", similarity_euclidean)


print("\n" + "=" * 60)
print("3.1.2 Manhattan (City Block) Distance & Similarity")
print("=" * 60)
pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Manhattan Distance
manhattan_dist = distance.cityblock(pointA, pointB)
print("Manhattan Distance:", manhattan_dist)

# Similarity (inverse of distance)
similarity_manhattan = 1 / (1 + manhattan_dist)
print("Manhattan Similarity:", similarity_manhattan)


print("\n" + "=" * 60)
print("3.1.3 Minkowski Distance (generalized form, p=3) & Similarity")
print("=" * 60)
pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Minkowski Distance with p=3
minkowski_dist_p3 = distance.minkowski(pointA, pointB, p=3)
print("Minkowski Distance (p=3):", minkowski_dist_p3)

# Similarity (inverse of distance)
similarity_minkowski = 1 / (1 + minkowski_dist_p3)
print("Minkowski Similarity (p=3):", similarity_minkowski)


# ----------------------------------------------------
# 3.2 Statistical Measures
# ----------------------------------------------------
print("\n" + "=" * 60)
print("3.2.1 Pearson Correlation")
print("=" * 60)
import pandas as pd

# Example dataset
df_pearson = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 24, 33, 45, 60]
})

# Pearson correlation matrix
corr_matrix = df_pearson.corr(method='pearson')
print("Pearson Correlation Matrix:\n", corr_matrix)


print("\n" + "=" * 60)
print("3.2.2 Spearman Rank Correlation")
print("=" * 60)
from scipy.stats import spearmanr

# Example dataset
df_spearman = pd.DataFrame({
    'X': [10, 20, 30, 40, 50],
    'Y': [12, 18, 33, 47, 55]
})

# Spearman correlation coefficient and p-value
corr_value, p_value = spearmanr(df_spearman['X'], df_spearman['Y'])
print(f"Spearman Correlation Coefficient: {corr_value}")
print(f"P-value: {p_value}")


# ----------------------------------------------------
# 3.3 Edit-Based Measures (String/Text Data)
# ----------------------------------------------------
print("\n" + "=" * 60)
print("3.3 Hamming Distance")
print("=" * 60)
def hamming_distance(str1, str2):
    # Ensure strings are of equal length
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")

    # Count differing positions
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))

# Example usage
s1 = "karolin"
s2 = "kathrin"

dist = hamming_distance(s1, s2)
print(f"Hamming Distance between '{s1}' and '{s2}': {dist}")


# ----------------------------------------------------
# 4. Token-Based Measures (Sets/Text)
# ----------------------------------------------------
print("\n" + "=" * 60)
print("4.1 Jaccard Index")
print("=" * 60)
def jaccard_index(str1, str2):
    set1, set2 = set(str1.split()), set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Example usage
s1 = "data science is fun"
s2 = "science makes data useful"

print("Jaccard Index:", jaccard_index(s1, s2))


print("\n" + "=" * 60)
print("4.2 Cosine Similarity")
print("=" * 60)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example sentences
s1 = "data science is fun"
s2 = "science makes data useful"

# Convert text to vector representation
vectorizer = CountVectorizer().fit([s1, s2])
vectors = vectorizer.transform([s1, s2])

# Compute cosine similarity
cos_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
print("Cosine Similarity:", cos_sim)


# ----------------------------------------------------
# 5. Sequence-Based Measures
# ----------------------------------------------------
print("\n" + "=" * 60)
print("5. Longest Common Subsequence (LCS)")
print("=" * 60)
def lcs_length(X, Y):
    m, n = len(X), len(Y)
    # Create a matrix to store lengths of LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Build the matrix
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

# Example usage
seq1 = "ABCDEF"
seq2 = "AEBDF"
length = lcs_length(seq1, seq2)
print(f"Longest Common Subsequence length between '{seq1}' and '{seq2}': {length}")
print("=" * 60)
