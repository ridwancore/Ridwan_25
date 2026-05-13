import pandas as pd
from feature_extraction import extract_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

# Load dataset
data = pd.read_csv("urls.csv")

# Dataset must contain:
# url,label

X = []
y = []

for url in data['url']:
    X.append(extract_features(url))

y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "phishing_model.pkl")

print("Model saved successfully!")