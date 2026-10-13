# Stage 4: Data Modeling

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("=" * 50)
print("STAGE 4: DATA MODELING")
print("=" * 50)

# Load and preprocess dataset
df = sns.load_dataset("titanic")
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop_duplicates(inplace=True)
df = pd.get_dummies(df, columns=['sex', 'class', 'embarked'], drop_first=True)
df['family_size'] = df['sibsp'] + df['parch']

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

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")
