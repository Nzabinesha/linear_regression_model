# Medicine Price Prediction API

FastAPI service that wraps the trained model from `../linear_regression/models/best_model.pkl`.

## Endpoints
- `GET /` — health check
- `POST /predict` — predict a medicine's price. Body: JSON matching `MedicineFeatures` (see `/docs` for the full schema, types, and valid ranges/categories).
- `POST /retrain` — upload a CSV (same columns as the cleaned training set + `price`) to retrain and hot-swap the model.

## Run locally
```bash
cd summative/API
uv run uvicorn prediction:app --reload
# Swagger UI at http://127.0.0.1:8000/docs
```

## Example request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "chemical_class": "OTHER"
  }'
```

## Deploying to Render
1. Push this repo to GitHub.
2. On Render: **New +** → **Web Service** → connect the repo.
3. Root directory: `summative/API`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn prediction:app --host 0.0.0.0 --port $PORT`
6. Once deployed, your Swagger UI is at: `https://<your-service>.onrender.com/docs`
7. Put that URL in the root `README.md`.

## CORS configuration
See the comment block above `CORSMiddleware` in `prediction.py`. Origins are
explicitly allow-listed (not `*`) and restricted to `GET`/`POST` — update
`ALLOWED_ORIGINS` with your actual deployed Flutter web origin (and/or your
Render URL for same-origin testing) before recording the demo video.
- `POST /upload-data` — drop a new labeled CSV into the incoming-data queue. Does not retrain by itself.
- `GET /retrain-status` — check watcher status: last check time, last retrain time, rows used, resulting RMSE, files queued.
