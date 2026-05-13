import joblib
from feature_extraction import extract_features

# Load trained model
model = joblib.load("phishing_model.pkl")

# User input
url = input("Enter URL: ")

# Extract features
features = extract_features(url)

# Prediction
prediction = model.predict([features])[0]

if prediction == 1:
    print("⚠️ Phishing URL Detected")
else:
    print("✅ Legitimate URL")