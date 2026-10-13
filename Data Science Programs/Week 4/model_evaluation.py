# Stage 5: Model Evaluation

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

print("=" * 50)
print("STAGE 5: MODEL EVALUATION")
print("=" * 50)

# Load and preprocess dataset
df = sns.load_dataset("titanic")
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop_duplicates(inplace=True)
df = pd.get_dummies(df, columns=['sex', 'class', 'embarked'], drop_first=True)
df['family_size'] = df['sibsp'] + df['parch']

# Features and target
X = df.drop(columns=['survived'])
y = df['survived']
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median(numeric_only=True))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)
