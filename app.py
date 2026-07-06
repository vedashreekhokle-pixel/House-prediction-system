"""
app.py
-------
Flask web application that serves the house price prediction model.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "house_price_model.pkl")

app = Flask(__name__)

# Load model once at startup
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

LOCATIONS = ["Downtown", "Suburb", "Rural", "Uptown", "Industrial"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", locations=LOCATIONS, prediction=None, form_data=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            locations=LOCATIONS,
            prediction=None,
            error="Model not found. Please run 'python train_model.py' first.",
            form_data=None,
        )

    try:
        form_data = {
            "area_sqft": float(request.form["area_sqft"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "stories": int(request.form["stories"]),
            "garage": int(request.form["garage"]),
            "age_years": int(request.form["age_years"]),
            "distance_to_city_km": float(request.form["distance_to_city_km"]),
            "school_rating": int(request.form["school_rating"]),
            "crime_rate": float(request.form["crime_rate"]),
            "location": request.form["location"],
        }

        input_df = pd.DataFrame([form_data])
        prediction = model.predict(input_df)[0]
        prediction = round(float(prediction), 2)

        return render_template(
            "index.html",
            locations=LOCATIONS,
            prediction=prediction,
            form_data=form_data,
            error=None,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            locations=LOCATIONS,
            prediction=None,
            form_data=None,
            error=f"Something went wrong: {exc}",
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint, e.g. for programmatic use / Postman / curl."""
    if model is None:
        return jsonify({"error": "Model not found. Run train_model.py first."}), 500

    data = request.get_json(force=True)
    required = [
        "area_sqft", "bedrooms", "bathrooms", "stories", "garage",
        "age_years", "distance_to_city_km", "school_rating", "crime_rate", "location",
    ]
    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(float(prediction), 2)})


if __name__ == "__main__":
    app.run(debug=True)
