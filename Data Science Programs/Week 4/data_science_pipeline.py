"""
Week 4: Implement & Analyze Data Science Workflow and Processing Pipeline Stages

This script executes all stages of the Data Science Workflow and Processing Pipeline:
1. Problem Definition
2. Data Collection, Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Data Modeling
5. Model Evaluation
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

output_dir = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# STAGE 1: PROBLEM DEFINITION
# ==============================================================================
print("=" * 70)
print("STAGE 1: PROBLEM DEFINITION")
print("=" * 70)

# Define the problem:
# Predict whether a passenger survived the Titanic disaster based on features.
objective = "Classification: Survived (Yes/No)"
success_criteria = "Accuracy > 80%"
constraints = "Limited features, missing values, imbalanced classes"

print("Objective:", objective)
print("Success Criteria:", success_criteria)
print("Constraints:", constraints)


# ==============================================================================
# STAGE 2: DATA COLLECTION, DATA CLEANING & PREPROCESSING
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 2: DATA COLLECTION, DATA CLEANING & PREPROCESSING")
print("=" * 70)

# Data Collection
# Load dataset (Titanic dataset from seaborn or CSV)
df = sns.load_dataset("titanic")
print("Data shape:", df.shape)
print("\nFirst 5 rows of raw dataset:")
print(df.head())

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop duplicates
df.drop_duplicates(inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'class', 'embarked'], drop_first=True)

# Feature engineering: family size
df['family_size'] = df['sibsp'] + df['parch']

print("\nData shape after cleaning & feature engineering:", df.shape)
print("\nFirst 5 rows of preprocessed dataset:")
print(df.head())


# ==============================================================================
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# 1. Histogram of age
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plot1_path = os.path.join(output_dir, "age_distribution.png")
plt.savefig(plot1_path, bbox_inches='tight')
print(f"Age distribution plot saved as: {plot1_path}")
plt.close()

# 2. Correlation matrix
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plot2_path = os.path.join(output_dir, "correlation_matrix.png")
plt.savefig(plot2_path, bbox_inches='tight')
print(f"Correlation matrix heatmap saved as: {plot2_path}")
plt.close()


# ==============================================================================
# STAGE 4: DATA MODELING
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 4: DATA MODELING")
print("=" * 70)

# Define features and target
X = df.drop(columns=['survived'])
y = df['survived']

# Convert remaining categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
X = X.fillna(X.median(numeric_only=True))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained successfully!")


# ==============================================================================
# STAGE 5: MODEL EVALUATION
# ==============================================================================
print("\n" + "=" * 70)
print("STAGE 5: MODEL EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Check success criteria
if accuracy > 0.80:
    print(f"\n[SUCCESS] Model accuracy ({accuracy*100:.2f}%) exceeds the success criteria (> 80%)!")
else:
    print(f"\n[INFO] Model accuracy ({accuracy*100:.2f}%) did not meet the > 80% criteria.")
