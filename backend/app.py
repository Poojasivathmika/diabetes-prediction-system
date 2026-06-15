from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)

# Allow frontend requests from anywhere
CORS(app)

# Load model and scaler
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Home route
@app.route('/')
def home():
    return jsonify({
        "message": "Diabetes Prediction API Running Successfully"
    })

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigree']),
            float(data['age'])
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = int(model.predict(features_scaled)[0])

        probability = float(
            model.predict_proba(features_scaled)[0][1]
        )

        return jsonify({
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk": "High Risk" if prediction == 1 else "Low Risk"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)