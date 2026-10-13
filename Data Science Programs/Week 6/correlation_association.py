import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import pearsonr, spearmanr, chi2_contingency

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("--- Correlation & Association Techniques ---")

# 1. Pearson Correlation (Linear relationship between continuous variables)
print("\n1. Pearson Correlation Matrix (Linear Relationship):")
pearson_corr = df.corr(method='pearson')
print(pearson_corr)

# 2. Spearman Rank Correlation (Non-linear monotonic relationship)
print("\n2. Spearman Rank Correlation Matrix (Monotonic Relationship):")
spearman_corr = df.corr(method='spearman')
print(spearman_corr)

# 3. Chi-Square Test for Association (Categorical variables)
print("\n3. Chi-Square Test of Independence (Categorical Association):")
# Example survey data: Customer Preference vs Gender
sample_cat_data = pd.DataFrame({
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Preference': ['Product A', 'Product B', 'Product A', 'Product B', 'Product A', 'Product A', 'Product B', 'Product B', 'Product A', 'Product A', 'Product B', 'Product B']
})

contingency_table = pd.crosstab(sample_cat_data['Gender'], sample_cat_data['Preference'])
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print("Contingency Table:")
print(contingency_table)
print(f"\nChi-Square Statistic : {chi2_stat:.4f}")
print(f"p-value               : {p_val:.4f}")
print(f"Degrees of Freedom    : {dof}")
print("Expected Frequencies Table:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
