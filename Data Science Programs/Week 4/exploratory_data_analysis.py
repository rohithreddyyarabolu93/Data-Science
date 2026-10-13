# Stage 3: Exploratory Data Analysis (EDA)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("=" * 50)
print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# Load and prepare data
df = sns.load_dataset("titanic")
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop_duplicates(inplace=True)
df = pd.get_dummies(df, columns=['sex', 'class', 'embarked'], drop_first=True)
df['family_size'] = df['sibsp'] + df['parch']

# Output directory for saving plots
output_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Histogram of age
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plot1_path = os.path.join(output_dir, "age_distribution.png")
plt.savefig(plot1_path, bbox_inches='tight')
print(f"Age distribution plot saved to: {plot1_path}")
plt.show()

# 2. Correlation matrix
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plot2_path = os.path.join(output_dir, "correlation_matrix.png")
plt.savefig(plot2_path, bbox_inches='tight')
print(f"Correlation matrix plot saved to: {plot2_path}")
plt.show()
