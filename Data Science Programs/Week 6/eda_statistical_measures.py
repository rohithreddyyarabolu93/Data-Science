"""
Week 6: Implement & Analyze Exploratory Data Analysis (EDA) Techniques Using Statistical Measures
==================================================================================================
This script provides a consolidated analysis covering:
1. Measures of Central Tendency (Mean, Median, Mode)
2. Measures of Dispersion (Range, Variance, Standard Deviation, IQR)
3. Shape of Distribution (Skewness, Kurtosis)
4. Correlation & Association (Pearson, Spearman, Chi-Square Test)
5. Hypothesis Testing (T-test, ANOVA, Z-test)
6. Outlier Detection (Z-score, IQR, Grubbs' Test)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris

# ==========================================
# Load Iris Dataset
# ==========================================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=" * 70)
print("WEEK 6: EXPLORATORY DATA ANALYSIS USING STATISTICAL MEASURES")
print("=" * 70)
print("\nFirst 5 rows of Iris dataset:")
print(df.head())

# ==========================================
# 1. Measures of Central Tendency
# ==========================================
print("\n" + "-" * 40)
print("1. MEASURES OF CENTRAL TENDENCY")
print("-" * 40)
numeric_df = df[iris.feature_names]

print("\nMean Values (Average):")
print(numeric_df.mean())

print("\nMedian Values (Middle Value):")
print(numeric_df.median())

print("\nMode Values (Most Frequent Value):")
print(numeric_df.mode().iloc[0])

# ==========================================
# 2. Measures of Dispersion
# ==========================================
print("\n" + "-" * 40)
print("2. MEASURES OF DISPERSION")
print("-" * 40)

range_values = numeric_df.max() - numeric_df.min()
print("\nRange (Max - Min):")
print(range_values)

variance_values = numeric_df.var()
print("\nVariance:")
print(variance_values)

std_values = numeric_df.std()
print("\nStandard Deviation:")
print(std_values)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
print("\nInterquartile Range (IQR = Q3 - Q1):")
print(IQR)

# ==========================================
# 3. Shape of Distribution
# ==========================================
print("\n" + "-" * 40)
print("3. SHAPE OF DISTRIBUTION")
print("-" * 40)

print("\nSkewness (Asymmetry):")
print(numeric_df.skew())

print("\nKurtosis (Peakedness / Tail Heaviness):")
print(numeric_df.kurt())

# ==========================================
# 4. Correlation & Association
# ==========================================
print("\n" + "-" * 40)
print("4. CORRELATION & ASSOCIATION")
print("-" * 40)

print("\nPearson Correlation Matrix (Linear Relationships):")
print(numeric_df.corr(method='pearson'))

print("\nSpearman Rank Correlation Matrix (Monotonic Relationships):")
print(numeric_df.corr(method='spearman'))

# Chi-Square Test for Association
sample_cat = pd.DataFrame({
    'Category_A': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Category_B': ['X', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'Y', 'X', 'Y']
})
contingency_tab = pd.crosstab(sample_cat['Category_A'], sample_cat['Category_B'])
chi2, p_val, dof, _ = stats.chi2_contingency(contingency_tab)
print("\nChi-Square Test of Association:")
print(f"Chi2 Stat: {chi2:.4f}, p-value: {p_val:.4f}, Degrees of Freedom: {dof}")

# ==========================================
# 5. Hypothesis Testing
# ==========================================
print("\n" + "-" * 40)
print("5. HYPOTHESIS TESTING")
print("-" * 40)

# Two-sample t-test
setosa_sl = df[df['species_name'] == 'setosa']['sepal length (cm)']
versicolor_sl = df[df['species_name'] == 'versicolor']['sepal length (cm)']
virginica_sl = df[df['species_name'] == 'virginica']['sepal length (cm)']

t_stat, t_p = stats.ttest_ind(setosa_sl, versicolor_sl)
print(f"Two-sample T-test (Setosa vs Versicolor Sepal Length): T = {t_stat:.4f}, p-value = {t_p:.4e}")

# One-way ANOVA
f_stat, f_p = stats.f_oneway(setosa_sl, versicolor_sl, virginica_sl)
print(f"One-Way ANOVA (Sepal Length across 3 species)       : F = {f_stat:.4f}, p-value = {f_p:.4e}")

# One-sample Z-test
sample_vals = df['sepal length (cm)'].values
pop_mean, pop_std = 5.8, 0.8
z_stat = (np.mean(sample_vals) - pop_mean) / (pop_std / np.sqrt(len(sample_vals)))
z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"One-Sample Z-Test (Sepal Length vs Pop Mean 5.8)    : Z = {z_stat:.4f}, p-value = {z_p:.4f}")

# ==========================================
# 6. Outlier Detection
# ==========================================
print("\n" + "-" * 40)
print("6. OUTLIER DETECTION")
print("-" * 40)

test_data = pd.Series([10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 100, -50])
print(f"Sample Series: {test_data.tolist()}")

# Z-score Outliers (|Z| > 3)
z_scores = np.abs(stats.zscore(test_data))
outliers_z = test_data[z_scores > 3].tolist()
print(f"Z-score Method Outliers (|Z| > 3): {outliers_z}")

# IQR Outliers
q1 = test_data.quantile(0.25)
q3 = test_data.quantile(0.75)
iqr = q3 - q1
outliers_iqr = test_data[(test_data < (q1 - 1.5 * iqr)) | (test_data > (q3 + 1.5 * iqr))].tolist()
print(f"IQR Method Outliers (outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]): {outliers_iqr}")

print("\n" + "=" * 70)
print("ALL WEEK 6 EDA TECHNIQUES SUCCESSFULLY EXECUTED")
print("=" * 70)
