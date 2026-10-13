import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris

print("--- Hypothesis Testing Methods ---")

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 1. Two-sample T-test (Compare means between 2 groups)
# Comparing sepal length between Setosa and Versicolor
setosa_sepal = df[df['species_name'] == 'setosa']['sepal length (cm)']
versicolor_sepal = df[df['species_name'] == 'versicolor']['sepal length (cm)']

t_stat, t_pval = stats.ttest_ind(setosa_sepal, versicolor_sepal)
print("\n1. Independent Two-Sample T-test:")
print(f"Comparison: Setosa mean ({setosa_sepal.mean():.2f}) vs Versicolor mean ({versicolor_sepal.mean():.2f})")
print(f"T-statistic : {t_stat:.4f}")
print(f"p-value     : {t_pval:.4e}")
print(f"Conclusion  : {'Reject null hypothesis (significant difference)' if t_pval < 0.05 else 'Fail to reject null hypothesis'}")

# 2. ANOVA (Analysis of Variance - Compare means across 3+ groups)
# Comparing sepal length across all 3 Iris species
virginica_sepal = df[df['species_name'] == 'virginica']['sepal length (cm)']

f_stat, f_pval = stats.f_oneway(setosa_sepal, versicolor_sepal, virginica_sepal)
print("\n2. One-Way ANOVA (Comparison across all 3 species):")
print(f"F-statistic : {f_stat:.4f}")
print(f"p-value     : {f_pval:.4e}")
print(f"Conclusion  : {'Reject null hypothesis (significant difference between group means)' if f_pval < 0.05 else 'Fail to reject null hypothesis'}")

# 3. Z-test (Compare sample mean to population mean)
# Testing if sample mean differs from hypothesized population mean
sample = df['sepal length (cm)'].values
pop_mean = 5.8
pop_std = 0.8
n = len(sample)
sample_mean = np.mean(sample)

z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
z_pval = 2 * (1 - stats.norm.cdf(abs(z_stat))) # Two-tailed test

print("\n3. One-Sample Z-test:")
print(f"Sample Mean : {sample_mean:.4f}, Hypothesized Population Mean: {pop_mean}")
print(f"Z-statistic : {z_stat:.4f}")
print(f"p-value     : {z_pval:.4f}")
print(f"Conclusion  : {'Reject null hypothesis' if z_pval < 0.05 else 'Fail to reject null hypothesis (no significant difference)'}")
