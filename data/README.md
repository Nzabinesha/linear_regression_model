# Medicine Price Prediction API

FastAPI service that wraps the trained model from `../linear_regression/models/best_model.pkl`.

## Endpoints
- `GET /` — health check, also shows last auto-retrain time and count
- `POST /predict` — predict a medicine's price. Body: JSON matching `MedicineFeatures` (see `/docs` for the full schema, types, and valid ranges/categories).
- `POST /upload-data` — drop a new labeled CSV into the incoming-data queue. **Does not retrain by itself** — see "Automatic retraining" below.
- `GET /retrain-status` — check watcher status: last check time, last retrain time, rows used, resulting test RMSE, how many files are currently queued.
- `POST /retrain` — manual/instant retrain (admin fallback for quick testing; not the graded mechanism).

## Automatic ("reactive") retraining

Instead of a human calling `/retrain`, a background thread starts when the
API boots and polls `data/incoming/` every `RETRAIN_CHECK_INTERVAL_SECONDS`
seconds (default 20s, override via env var). Any CSV that lands there —
via `POST /upload-data`, scp'd directly onto the server, or written by an
upstream pipeline — is picked up **on its own**: the watcher retrains a
fresh Random Forest on it, hot-swaps the running model, and archives the
file to `data/processed/`. Nobody has to trigger anything by hand.

To demo this:
```bash
# 1. Queue a new data file
curl -X POST http://127.0.0.1:8000/upload-data -F "file=@new_medicines.csv"

# 2. Wait ~20s (or however long RETRAIN_CHECK_INTERVAL_SECONDS is set to), then check:
curl http://127.0.0.1:8000/retrain-status
# -> last_retrain_time and total_auto_retrains will have updated on their own
```

**Known limitation (worth mentioning in the video):** on Render's free tier,
local disk is not guaranteed to persist across deploys/restarts, so this is
a demo-appropriate implementation of "reactive retraining," not a
production-grade one. A production version would point `INCOMING_DIR` at
durable storage (S3, a mounted persistent disk, or a database table) so
queued files survive restarts.

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
