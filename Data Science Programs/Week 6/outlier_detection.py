import numpy as np
import pandas as pd
from scipy import stats

print("--- Outlier Detection Methods ---")

# Sample data containing realistic data with deliberate outliers
np.random.seed(42)
data = np.array([12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 105, -45])
df = pd.DataFrame(data, columns=['Values'])

print("Sample Dataset:")
print(df['Values'].values)

# 1. Z-score Method: Flag values beyond +/- 3 standard deviations (|z| > 3)
z_scores = stats.zscore(df['Values'])
df['Z_score'] = z_scores
outliers_zscore = df[np.abs(df['Z_score']) > 3]

print("\n1. Z-Score Method (|Z| > 3):")
print(f"Mean: {df['Values'].mean():.2f}, Std Dev: {df['Values'].std():.2f}")
if not outliers_zscore.empty:
    print("Detected Outliers (|Z| > 3):")
    print(outliers_zscore[['Values', 'Z_score']])
else:
    print("No outliers detected with strict threshold |Z| > 3.")

# Also checking threshold |Z| > 2 for smaller sample sizes
outliers_z2 = df[np.abs(df['Z_score']) > 2]
print("Outliers with (|Z| > 2):")
print(outliers_z2[['Values', 'Z_score']])

# 2. IQR Method: Identify values outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
Q1 = df['Values'].quantile(0.25)
Q3 = df['Values'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['Values'] < lower_bound) | (df['Values'] > upper_bound)]

print("\n2. Interquartile Range (IQR) Method:")
print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR                 : {IQR:.2f}")
print(f"Lower Bound (Q1 - 1.5*IQR): {lower_bound:.2f}")
print(f"Upper Bound (Q3 + 1.5*IQR): {upper_bound:.2f}")
print("Detected Outliers:")
print(outliers_iqr[['Values']])

# 3. Grubbs' Test: Detect single outliers in normally distributed data
def grubbs_test(values, alpha=0.05):
    n = len(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    
    # Absolute deviations from mean
    abs_deviations = np.abs(values - mean_val)
    max_idx = np.argmax(abs_deviations)
    suspected_val = values[max_idx]
    
    # Calculate G statistic
    G_calc = abs_deviations[max_idx] / std_val
    
    # Critical value of G using t-distribution
    t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    
    is_outlier = G_calc > G_crit
    return suspected_val, G_calc, G_crit, is_outlier

suspected, G_calc, G_crit, is_outlier = grubbs_test(df['Values'].values)

print("\n3. Grubbs' Test for Single Outlier:")
print(f"Suspected Outlier Value : {suspected}")
print(f"Calculated G-Statistic  : {G_calc:.4f}")
print(f"Critical G-Value (a=0.05): {G_crit:.4f}")
print(f"Is statistically significant outlier? {is_outlier}")
