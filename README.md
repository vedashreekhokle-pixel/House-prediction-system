# 🏠 House Price Prediction System

A complete, ready-to-run machine learning project that predicts house prices
based on property features (area, bedrooms, location, etc.) using a Random
Forest Regressor, served through a Flask web app.

## Project Structure

```
house_price_prediction/
├── app.py                  # Flask web app (UI + JSON API)
├── train_model.py          # Trains the model and saves it to model/
├── requirements.txt        # Python dependencies
├── data/
│   ├── generate_data.py    # Generates a synthetic housing dataset
│   └── house_data.csv      # Dataset (generated)
├── model/
│   └── house_price_model.pkl   # Trained model (generated)
├── templates/
│   └── index.html           # Web UI
└── static/
    └── style.css             # Styling
```

## 1. Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Generate the dataset (optional — auto-generated if missing)

```bash
python data/generate_data.py
```

This creates `data/house_data.csv` with 2000 synthetic house records
(area, bedrooms, bathrooms, stories, garage, age, distance to city,
school rating, crime rate, location, and price).

You can swap this out with your own real dataset — just make sure it has
the same column names, or update `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`
in `train_model.py` to match your columns.

## 3. Train the model

```bash
python train_model.py
```

This will:
- Load (or generate) the dataset
- Split it into train/test sets
- Train a `RandomForestRegressor` inside a scikit-learn `Pipeline`
  (with scaling + one-hot encoding built in)
- Print evaluation metrics (MAE, RMSE, R²)
- Save the trained pipeline to `model/house_price_model.pkl`

## 4. Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Fill in the property
details and click **Predict Price** to get an estimate.

## 5. JSON API (optional)

You can also call the model programmatically:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "area_sqft": 2200,
        "bedrooms": 4,
        "bathrooms": 3,
        "stories": 2,
        "garage": 2,
        "age_years": 5,
        "distance_to_city_km": 3.5,
        "school_rating": 8,
        "crime_rate": 1.5,
        "location": "Uptown"
      }'
```

Response:
```json
{ "predicted_price": 512340.75 }
```

## Using Your Own Real Dataset

To use a real dataset (e.g. from Kaggle):

1. Replace `data/house_data.csv` with your own CSV file (or point
   `DATA_PATH` in `train_model.py` to it).
2. Update `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` in `train_model.py`
   to match your dataset's columns.
3. Update the form fields in `templates/index.html` and `app.py` to match.
4. Re-run `python train_model.py`.

## Notes

- The bundled dataset is **synthetic** (randomly generated with a realistic
  pricing formula) so the project works out of the box without needing to
  download anything. Swap in a real dataset for production use.
- The model is a `RandomForestRegressor`; you can try other algorithms
  (Linear Regression, Gradient Boosting, XGBoost) by editing
  `build_pipeline()` in `train_model.py`.
- Retrain the model any time by re-running `python train_model.py` — it
  will overwrite `model/house_price_model.pkl`.
