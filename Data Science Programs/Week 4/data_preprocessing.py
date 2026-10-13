# Stage 2: Data Collection, Data Cleaning & Preprocessing

import pandas as pd
import seaborn as sns

print("=" * 50)
print("STAGE 2: DATA COLLECTION & PREPROCESSING")
print("=" * 50)

# Data Collection
# Load dataset (Titanic dataset from seaborn or CSV)
df = sns.load_dataset("titanic")
print("Data shape:", df.shape)
print("\nInitial Data Head:")
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

print("\nProcessed Data Shape after Cleaning & Feature Engineering:", df.shape)
print("\nProcessed Data Head:")
print(df.head())
