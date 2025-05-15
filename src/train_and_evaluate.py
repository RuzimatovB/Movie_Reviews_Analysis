import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import processed features and labels from previous step
from preprocess_and_vectorize import X, y

# 1. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Predict on the test set
y_pred = model.predict(X_test)

# 4. Evaluate the model
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
