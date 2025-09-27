"""
Crop Recommendation Model Training Script

This script trains a Random Forest classifier to recommend crops based on
soil and environmental parameters. It preprocesses the data, trains the model,
and saves the necessary components for deployment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the crop recommendation dataset
crop = pd.read_csv("Crop_recommendation.csv")

# Separate features (X) and target (y)
X = crop.drop('label', axis=1)  # Features: N, P, K, temperature, humidity, ph, rainfall
y = crop['label']  # Target: crop names

# Encode categorical labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features using StandardScaler (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model performance on test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save trained model, scaler, and label encoder for deployment
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le, open('labelencoder.pkl', 'wb'))

print("Model, scaler, and label encoder saved successfully.")