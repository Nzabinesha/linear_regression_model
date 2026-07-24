"""
Loads the saved best-performing model and makes a prediction for a single
medicine, given its engineered feature values. This is the function that
Task 2's FastAPI endpoint (summative/API/prediction.py) wraps.

Run directly for a quick sanity check:
    uv run python predict.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"

NUMERIC_FEATURES = [
    "is_discontinued", "manufacturer_size", "pack_quantity",
    "composition1_strength_mg", "has_composition2", "num_substitutes",
    "num_side_effects", "num_uses", "habit_forming",
]
CATEGORICAL_FEATURES = ["pack_container", "pack_form", "therapeutic_class", "chemical_class"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(pipeline, features: dict) -> float:
    """
    features: dict with all keys in ALL_FEATURES.
    Returns the predicted price in Rs (already inverse-transformed from
    the log scale the model was trained on).
    """
    missing = [f for f in ALL_FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    row_df = pd.DataFrame([{f: features[f] for f in ALL_FEATURES}])
    log_price_pred = pipeline.predict(row_df)[0]
    return float(np.expm1(log_price_pred))


if __name__ == "__main__":
    model = load_model()

    example = {
        "is_discontinued": 0,
        "manufacturer_size": 12,
        "pack_quantity": 10.0,
        "composition1_strength_mg": 500.0,
        "has_composition2": 0,
        "num_substitutes": 5,
        "num_side_effects": 3,
        "num_uses": 1,
        "habit_forming": 0,
        "pack_container": "strip",
        "pack_form": "tablets",
        "therapeutic_class": "ANTI INFECTIVES",
        "chemical_class": "OTHER",
    }

    price = predict_price(model, example)
    print(f"Predicted price: Rs {price:.2f}")
