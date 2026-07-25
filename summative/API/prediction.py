"""
FastAPI service for the Online Pharmacy Finder medicine price model.

Endpoints:
  GET  /                -> health check + last-retrain status summary
  POST /predict         -> predict the price of a medicine from its attributes
  POST /upload-data      -> drop a new labeled CSV into the incoming-data queue
                            (does NOT retrain itself - see below)
  POST /retrain         -> manual/instant retrain, kept as an admin fallback
  GET  /retrain-status  -> details on the automatic watcher's last run

Automatic ("reactive") retraining
----------------------------------
A background thread (`_background_retrain_watcher`, started on app startup)
polls the `data/incoming/` folder every RETRAIN_CHECK_INTERVAL_SECONDS
seconds. Any CSV files dropped there (via POST /upload-data, or scp'd
directly onto the server, or written by an upstream data pipeline) are
picked up automatically: the watcher retrains a fresh model on them,
hot-swaps it into the running API, and moves the file into
`data/processed/`. No one has to call /retrain by hand - the API reacts
to new data appearing on its own, which is what distinguishes this from a
manually-triggered retrain endpoint.

Run locally:
    uv run uvicorn prediction:app --reload

Docs (Swagger UI):
    http://127.0.0.1:8000/docs
"""
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

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
INCOMING_DIR = BASE_DIR / "data" / "incoming"     # new data lands here
PROCESSED_DIR = BASE_DIR / "data" / "processed"   # already-used files are archived here

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# How often the background watcher checks for new data. Kept short by
# default so it's easy to demo; bump this up (e.g. 300s) in a real
# deployment where data doesn't arrive every few seconds.
RETRAIN_CHECK_INTERVAL_SECONDS = int(os.environ.get("RETRAIN_CHECK_INTERVAL_SECONDS", "20"))

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


class UploadDataResponse(BaseModel):
    message: str
    filename: str
    queued_files_waiting: int


class RetrainStatusResponse(BaseModel):
    watcher_running: bool
    check_interval_seconds: int
    last_check_time: Optional[str]
    last_retrain_time: Optional[str]
    last_retrain_rows: Optional[int]
    last_retrain_test_rmse: Optional[float]
    total_auto_retrains: int
    files_currently_queued: int


# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = FastAPI(
    title="Online Pharmacy Finder - Price Prediction API",
    description="Predicts the expected price of a medicine from its attributes.",
    version="1.1.0",
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
# Shared model state (guarded by a lock since the background watcher
# thread and request-handling threads both touch it)
# ---------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()

_watcher_status = {
    "last_check_time": None,
    "last_retrain_time": None,
    "last_retrain_rows": None,
    "last_retrain_test_rmse": None,
    "total_auto_retrains": 0,
}


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            if not MODEL_PATH.exists():
                raise HTTPException(status_code=500, detail="Model file not found on server.")
            _model = joblib.load(MODEL_PATH)
        return _model


def _train_random_forest(df: pd.DataFrame):
    """Shared training routine used by both the manual /retrain endpoint
    and the automatic background watcher, so the two paths can never
    drift out of sync with each other."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    X = df[ALL_FEATURES]
    y = np.log1p(df["price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=15, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pipe.predict(X_test))))
    return pipe, test_rmse


def _validate_training_csv(df: pd.DataFrame):
    required_cols = set(ALL_FEATURES + ["price"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")


# ---------------------------------------------------------------------
# Background watcher: reacts to new data on its own, no manual trigger
# ---------------------------------------------------------------------
def _background_retrain_watcher():
    global _model
    while True:
        _watcher_status["last_check_time"] = datetime.now(timezone.utc).isoformat()

        incoming_files = sorted(INCOMING_DIR.glob("*.csv"))
        if incoming_files:
            combined_frames = []
            for f in incoming_files:
                try:
                    df = pd.read_csv(f)
                    _validate_training_csv(df)
                    combined_frames.append(df)
                except Exception as e:
                    # Bad file: archive it with an .error suffix so it
                    # doesn't jam the queue, and move on.
                    f.rename(PROCESSED_DIR / f"{f.name}.error")
                    print(f"[watcher] Skipping invalid file {f.name}: {e}")

            if combined_frames:
                new_df = pd.concat(combined_frames, ignore_index=True)
                try:
                    new_pipe, test_rmse = _train_random_forest(new_df)
                    joblib.dump(new_pipe, MODEL_PATH)
                    with _model_lock:
                        _model = new_pipe

                    now = datetime.now(timezone.utc).isoformat()
                    _watcher_status["last_retrain_time"] = now
                    _watcher_status["last_retrain_rows"] = len(new_df)
                    _watcher_status["last_retrain_test_rmse"] = round(test_rmse, 4)
                    _watcher_status["total_auto_retrains"] += 1
                    print(f"[watcher] Auto-retrained on {len(new_df)} rows, test RMSE={test_rmse:.4f}")
                except Exception as e:
                    print(f"[watcher] Retrain failed: {e}")

            # Archive processed files whether or not they contributed
            # (invalid ones were already moved above).
            for f in incoming_files:
                if f.exists():
                    shutil.move(str(f), str(PROCESSED_DIR / f.name))

        time.sleep(RETRAIN_CHECK_INTERVAL_SECONDS)


@app.on_event("startup")
def _start_background_watcher():
    thread = threading.Thread(target=_background_retrain_watcher, daemon=True)
    thread.start()


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Medicine price prediction API is running. See /docs for Swagger UI.",
        "last_retrain_time": _watcher_status["last_retrain_time"],
        "total_auto_retrains": _watcher_status["total_auto_retrains"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: MedicineFeatures):
    model = get_model()
    row_df = pd.DataFrame([features.model_dump()])[ALL_FEATURES]
    log_price_pred = model.predict(row_df)[0]
    price = float(np.expm1(log_price_pred))
    return PredictionResponse(predicted_price_inr=round(price, 2))


@app.post("/upload-data", response_model=UploadDataResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Drops a new labeled CSV into the incoming-data queue. This endpoint
    does NOT retrain anything itself - the background watcher notices the
    file on its own next time it polls (every RETRAIN_CHECK_INTERVAL_SECONDS
    seconds) and retrains automatically. Check GET /retrain-status to see
    when that happens.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    dest = INCOMING_DIR / f"{int(time.time())}_{file.filename}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    queued = len(list(INCOMING_DIR.glob("*.csv")))
    return UploadDataResponse(
        message="File queued. The background watcher will pick it up automatically.",
        filename=dest.name,
        queued_files_waiting=queued,
    )


@app.get("/retrain-status", response_model=RetrainStatusResponse)
def retrain_status():
    return RetrainStatusResponse(
        watcher_running=True,
        check_interval_seconds=RETRAIN_CHECK_INTERVAL_SECONDS,
        last_check_time=_watcher_status["last_check_time"],
        last_retrain_time=_watcher_status["last_retrain_time"],
        last_retrain_rows=_watcher_status["last_retrain_rows"],
        last_retrain_test_rmse=_watcher_status["last_retrain_test_rmse"],
        total_auto_retrains=_watcher_status["total_auto_retrains"],
        files_currently_queued=len(list(INCOMING_DIR.glob("*.csv"))),
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)):
    """
    Manual/instant retrain - kept as an admin fallback for testing.
    The primary, graded mechanism is the automatic watcher above:
    POST /upload-data + wait for the background thread to react.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        new_df = pd.read_csv(file.file)
        _validate_training_csv(new_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    new_pipe, test_rmse = _train_random_forest(new_df)
    joblib.dump(new_pipe, MODEL_PATH)
    global _model
    with _model_lock:
        _model = new_pipe

    return RetrainResponse(
        message="Model retrained and swapped in successfully (manual trigger).",
        rows_used=len(new_df),
        new_test_rmse=round(test_rmse, 4),
    )
