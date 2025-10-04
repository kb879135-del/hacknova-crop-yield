from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# Load model
model = joblib.load("../models/crop_yield_model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "AI Crop Yield Prediction API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data["rainfall"], data["temperature"], data["soil_quality"]]])
    prediction = model.predict(features)[0]
    return jsonify({
        "input": data,
        "predicted_yield": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
