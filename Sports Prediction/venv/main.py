from flask import Flask, request, jsonify, render_template
from math import ceil
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm  # for z-score calculation
from sklearn.metrics import mean_absolute_error

app = Flask(__name__, static_folder='C:\\Users\\a\\OneDrive - Ashesi University\\Sports Prediction\\venv\\static')

# Load the pre-trained model and scaler
model = joblib.load("C:\\Users\\a\\OneDrive - Ashesi University\\Sports Prediction\\venv\\sports_prediction_model.pkl")
scaler = joblib.load("C:\\Users\\a\\OneDrive - Ashesi University\\Sports Prediction\\venv\\scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form and store them in a dictionary
    input_data = {key: int(value) for key, value in request.form.items()}
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    # Scale the input features
    scaled_features = scaler.transform(input_df)
    # Predict using the model
    prediction = model.predict(scaled_features)[0]
    output = ceil(prediction)
    
    return render_template('index.html', prediction_text=f"The estimated rating of the player entered is {output} with a confidence score of 89.99% ≈ 90%")

if __name__ == '__main__':
    app.run(debug=True)
