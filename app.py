"""
Crop Recommendation System - Flask Web Application

This application uses a machine learning model to recommend the best crop
based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall.
"""

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained model, scaler, and label encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("labelencoder.pkl", "rb"))

@app.route("/")
def index():
    """Render the main page with the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests.

    Extracts form data, preprocesses it, makes prediction, and returns result.
    """
    try:
        # Extract input values from form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Prepare features array
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

        # Scale features using pre-fitted scaler
        scaled = scaler.transform(features)

        # Make prediction
        pred = model.predict(scaled)[0]

        # Decode prediction to crop name
        crop = le.inverse_transform([pred])[0]

        # Format result message
        result = f"✅ {crop} is the best crop to be cultivated right there!"

    except Exception as e:
        # Handle any errors gracefully
        result = f"⚠️ Error: {e}"

    # Render template with result
    return render_template("index.html", result=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port, debug=True)

