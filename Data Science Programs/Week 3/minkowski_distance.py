import numpy as np
from scipy.spatial import distance

pointA = np.array([2, 4, 6])
pointB = np.array([5, 1, 9])

# Minkowski Distance with p=3
minkowski_dist_p3 = distance.minkowski(pointA, pointB, p=3)
print("Minkowski Distance (p=3):", minkowski_dist_p3)

# Similarity (inverse of distance)
similarity_minkowski = 1 / (1 + minkowski_dist_p3)
print("Minkowski Similarity (p=3):", similarity_minkowski)
