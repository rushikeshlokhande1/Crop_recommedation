# Crop Recommendation System

A machine learning-based web application that recommends the best crop to cultivate based on soil nutrients and environmental conditions.

## Features

- **Accurate Predictions**: Uses Random Forest classifier trained on comprehensive crop dataset
- **User-Friendly Interface**: Clean, responsive web interface built with Flask and Bootstrap
- **Real-time Recommendations**: Instant crop suggestions based on input parameters

## Input Parameters

- Nitrogen (N) - ppm
- Phosphorus (P) - ppm
- Potassium (K) - ppm
- Temperature - °C
- Humidity - %
- pH Value
- Rainfall - mm

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install flask scikit-learn pandas numpy
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. Start the web application:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser

## Project Structure

- `app.py` - Flask web application
- `train_model.py` - Model training script
- `Crop_recommendation.csv` - Dataset
- `templates/index.html` - Web interface
- `model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `labelencoder.pkl` - Label encoder
- `README.md` - Project documentation

## Model Performance

- Algorithm: Random Forest Classifier
- Accuracy: 99.32%
- Features: Standard scaled soil and environmental parameters

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Bootstrap
- HTML/CSS/JavaScript