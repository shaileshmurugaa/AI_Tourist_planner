import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# ------------------------------
# Load Hybrid Models
# ------------------------------
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
rule_engine = joblib.load("rule_engine.pkl")

print("✅ Models loaded successfully")

# ------------------------------
# Home Route
# ------------------------------
@app.route("/")
def home():
    return "🌍 AI Tourist Planner API is Running!"

# ------------------------------
# Prediction Route
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Example expected inputs
        temperature = float(data["temperature"])
        rainfall = float(data["rainfall"])
        humidity = float(data["humidity"])
        air_quality = float(data["air_quality"])
        budget = float(data["budget"])

        # Prepare input for ML models
        features = np.array([[temperature, rainfall, humidity, air_quality, budget]])

        # Model Predictions
        rf_pred = rf_model.predict(features)[0]
        xgb_pred = xgb_model.predict(features)[0]
        cluster = kmeans_model.predict(features)[0]

        # Hybrid Decision Logic
        hybrid_score = (rf_pred + xgb_pred) / 2

        # Rule Engine Suggestion
        suggestion = rule_engine.get(cluster, "General Travel")

        result = {
            "hybrid_score": round(float(hybrid_score), 2),
            "cluster": int(cluster),
            "recommendation": suggestion
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------------------
# Run App (Render Compatible)
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
