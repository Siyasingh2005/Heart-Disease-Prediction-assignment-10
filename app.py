"""
app.py
------
Task 3: API Development

A Flask REST API that:
 - Loads the trained heart-disease-prediction model.
 - Accepts patient clinical details as JSON input.
 - Returns the prediction as JSON.
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load trained model and expected feature order at startup
MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_order.pkl"

model = joblib.load(MODEL_PATH)
FEATURE_ORDER = joblib.load(FEATURES_PATH)


@app.route("/", methods=["GET"])
def home():
    """Simple landing page confirming the API is live."""
    return render_template("index.html", features=FEATURE_ORDER)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (useful for Render + evaluators)."""
    return jsonify({"status": "ok", "message": "Heart Disease Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON input with patient clinical parameters and returns
    a heart disease risk prediction.

    Example request body:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    try:
        data = request.get_json(force=True)

        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {missing}"
            }), 400

        # Build a single-row DataFrame in the exact training feature order
        input_df = pd.DataFrame([[data[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "risk_probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
