from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Crop Yield Prediction API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    return jsonify({
        "input": data,
        "predicted_yield": "Dummy prediction (model will come later)"
    })

if __name__ == "__main__":
    app.run(debug=True)

