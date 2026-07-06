"""
generate_data.py
-----------------
Generates a synthetic house-price dataset and saves it as data/house_data.csv.

Run this file directly if you want to regenerate the dataset:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
import os

def generate_house_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Feature generation -------------------------------------------------
    area_sqft = rng.integers(500, 6000, n_samples)                 # house area
    bedrooms = rng.integers(1, 7, n_samples)                        # number of bedrooms
    bathrooms = rng.integers(1, 5, n_samples)                       # number of bathrooms
    stories = rng.integers(1, 4, n_samples)                         # number of floors
    garage = rng.integers(0, 3, n_samples)                          # garage capacity (cars)
    age_years = rng.integers(0, 60, n_samples)                      # age of the house
    distance_to_city_km = np.round(rng.uniform(0.5, 40, n_samples), 2)
    school_rating = rng.integers(1, 11, n_samples)                  # 1 (poor) - 10 (excellent)
    crime_rate = np.round(rng.uniform(0, 10, n_samples), 2)         # 0 (safe) - 10 (high crime)

    locations = ["Downtown", "Suburb", "Rural", "Uptown", "Industrial"]
    location = rng.choice(locations, n_samples, p=[0.25, 0.35, 0.15, 0.2, 0.05])

    location_premium = {
        "Downtown": 45000,
        "Uptown": 30000,
        "Suburb": 10000,
        "Rural": -15000,
        "Industrial": -25000,
    }

    # ---- Price formula (with noise) ----------------------------------------
    base_price = (
        area_sqft * 120
        + bedrooms * 8000
        + bathrooms * 6000
        + stories * 5000
        + garage * 4000
        - age_years * 900
        - distance_to_city_km * 800
        + school_rating * 3500
        - crime_rate * 2200
    )

    price = base_price + np.array([location_premium[loc] for loc in location])
    noise = rng.normal(0, 15000, n_samples)
    price = np.maximum(price + noise, 25000)  # floor price so it never goes negative
    price = np.round(price, -2)  # round to nearest 100

    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "garage": garage,
        "age_years": age_years,
        "distance_to_city_km": distance_to_city_km,
        "school_rating": school_rating,
        "crime_rate": crime_rate,
        "location": location,
        "price": price,
    })

    return df


if __name__ == "__main__":
    df = generate_house_data()
    out_path = os.path.join(os.path.dirname(__file__), "house_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
