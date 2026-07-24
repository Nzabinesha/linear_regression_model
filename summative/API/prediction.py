"""
FastAPI service for the Online Pharmacy Finder medicine price model.

Endpoints:
  GET  /                -> health check
  POST /predict         -> predict the price of a medicine from its attributes
  POST /retrain         -> retrain the model on an uploaded CSV of new data
                           (manual/triggered retraining, per Task 2 requirement)

Run locally:
    uv run uvicorn prediction:app --reload

Docs (Swagger UI):
    http://127.0.0.1:8000/docs
"""
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

NUMERIC_FEATURES = [
    "is_discontinued", "manufacturer_size", "pack_quantity",
    "composition1_strength_mg", "has_composition2", "num_substitutes",
    "num_side_effects", "num_uses", "habit_forming",
]
CATEGORICAL_FEATURES = ["pack_container", "pack_form", "therapeutic_class", "chemical_class"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PackContainer = Literal["ampoule", "bottle", "box", "other", "packet", "strip", "tube", "vial"]
PackForm = Literal[
    "capsule sr", "capsules", "cream", "drop", "dry syrup", "eye drop", "gel",
    "infusion", "injection", "lotion", "ointment", "ophthalmic solution",
    "oral drops", "oral solution", "oral suspension", "other",
    "powder for injection", "soap", "soft gelatin capsules", "solution",
    "suspension", "syrup", "tablet", "tablet cr", "tablet dt", "tablet er",
    "tablet md", "tablet pr", "tablet sr", "tablets",
]
TherapeuticClass = Literal[
    "ANTI DIABETIC", "ANTI INFECTIVES", "ANTI MALARIALS", "ANTI NEOPLASTICS",
    "BLOOD RELATED", "CARDIAC", "DERMA", "GASTRO INTESTINAL", "GYNAECOLOGICAL",
    "HORMONES", "NEURO CNS", "OPHTHAL", "OPHTHAL OTOLOGICALS", "OTHERS",
    "OTOLOGICALS", "PAIN ANALGESICS", "RESPIRATORY",
    "SEX STIMULANTS REJUVENATORS", "STOMATOLOGICALS", "UNKNOWN", "UROLOGY",
    "VACCINES", "VITAMINS MINERALS NUTRIENTS",
]
ChemicalClass = Literal[
    "Aminoglycosides", "Aminopenicillins {Penicillins}", "Anabolic steroid",
    "Azole derivatives {Imidazoles}", "Azoles {Triazoles}",
    "Benzodiazepines Derivative",
    "Broad Spectrum (Third & fourth generation cephalosporins)",
    "Broad spectrum (Third & fourth generation cephalosporins}",
    "Carbazole Derivative", "Fluoroquinolone",
    "Gluco/mineralocorticoids, progestogins and derivatives",
    "Glucocorticoids",
    "Intermediate spectrum {Second generation cephalosporins}",
    "Macrolides", "OTHER", "P-Aminophenol Derivative",
    "Phenylacetic acid Derivative", "Piperazine Derivatives",
    "Pyrrole & heptanoic acid derivative", "Sulfinylbenzimidazole Derivative",
    "Timoprazole Derivative",
]


# ---------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------
class MedicineFeatures(BaseModel):
    is_discontinued: int = Field(..., ge=0, le=1, description="0 = active, 1 = discontinued")
    manufacturer_size: int = Field(..., ge=1, le=3000, description="Number of products listed by this manufacturer")
    pack_quantity: float = Field(..., gt=0, le=5000, description="Numeric pack size, e.g. 10 for a strip of 10 tablets")
    composition1_strength_mg: float = Field(..., ge=0, le=60000, description="Active ingredient strength in mg")
    has_composition2: int = Field(..., ge=0, le=1, description="1 if the medicine has a second active ingredient")
    num_substitutes: int = Field(..., ge=0, le=20, description="Number of listed substitute medicines")
    num_side_effects: int = Field(..., ge=0, le=50, description="Number of listed side effects")
    num_uses: int = Field(..., ge=0, le=10, description="Number of listed medical uses")
    habit_forming: int = Field(..., ge=0, le=1, description="1 if habit-forming")
    pack_container: PackContainer
    pack_form: PackForm
    therapeutic_class: TherapeuticClass
    chemical_class: ChemicalClass

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_inr: float


class RetrainResponse(BaseModel):
    message: str
    rows_used: int
    new_test_rmse: float


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(
    title="Online Pharmacy Finder - Price Prediction API",
    description="Predicts the expected price of a medicine from its attributes.",
    version="1.0.0",
)

# ---------------------------------------------------------------------
# CORS
#
# Reasoning:
# - allow_origins is an explicit allow-list (NOT "*") because this API
#   returns predictions that will be embedded in a specific Flutter web
#   build and a small set of known frontends; wildcarding origins would
#   let any third-party website call the endpoint from a user's browser.
# - allow_methods is restricted to GET/POST since those are the only
#   verbs this API exposes - no PUT/DELETE/PATCH is needed.
# - allow_headers includes Content-Type (JSON bodies) and Authorization
#   (reserved for future auth) rather than "*".
# - allow_credentials is False: the API is stateless and does not use
#   cookies/sessions, so there is no session credential to protect or leak.
# ---------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",       # local web/Flutter-web dev server
    "http://127.0.0.1:3000",
    "https://your-flutter-app-domain.example",  # replace with real deployed frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail="Model file not found on server.")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/")
def root():
    return {"status": "ok", "message": "Medicine price prediction API is running. See /docs for Swagger UI."}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MedicineFeatures):
    model = get_model()
    row_df = pd.DataFrame([features.model_dump()])[ALL_FEATURES]
    log_price_pred = model.predict(row_df)[0]
    price = float(np.expm1(log_price_pred))
    return PredictionResponse(predicted_price_inr=round(price, 2))


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """
    Triggers a retrain of the model using newly uploaded data.
    Expects a CSV with the same columns as the cleaned training dataset:
    all of ALL_FEATURES plus a 'price' column.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        new_df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required_cols = set(ALL_FEATURES + ["price"])
    missing = required_cols - set(new_df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Uploaded CSV is missing columns: {sorted(missing)}")

    # Retrain a fresh Random Forest pipeline (same architecture as the
    # notebook's best model) on the newly uploaded data.
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    X = new_df[ALL_FEATURES]
    y = np.log1p(new_df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    new_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=15, random_state=42)),
    ])
    new_pipe.fit(X_train, y_train)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, new_pipe.predict(X_test))))

    joblib.dump(new_pipe, MODEL_PATH)
    global _model
    _model = new_pipe  # hot-swap the in-memory model immediately

    return RetrainResponse(
        message="Model retrained and swapped in successfully.",
        rows_used=len(new_df),
        new_test_rmse=round(test_rmse, 4),
    )
