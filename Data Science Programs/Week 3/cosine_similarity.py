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
