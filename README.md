# Online Pharmacy Finder — Medicine Price Prediction

## Mission
An online pharmacy finder that helps people locate medicines without physically checking one pharmacy after another.
This model predicts a medicine's expected market price from its attributes, so the app can flag listings priced well above expected and point users to a better option instead of shopping around in person.

## Dataset
Two public Kaggle datasets (India), merged on medicine name:
- **A-Z Medicine Dataset of India** — name, price, manufacturer, pack size, composition, discontinued flag (~254k rows). https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india
- **250k Medicines Usage, Side Effects and Substitutes** — substitutes, side effects, uses, chemical/therapeutic class, habit-forming flag (~248k rows). https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes

After merging, cleaning, and clipping extreme price outliers: **~217,000 rows**.

## Links
- **Live API base URL:** https://linear-regression-model-2-5ni5.onrender.com
- **Swagger UI (interactive API docs):** https://linear-regression-model-2-5ni5.onrender.com/docs
- **YouTube video demo:** https://youtu.be/kOFmQL-GrU4

## Repo structure
```
linear_regression_model/
├── data/                              # place the two source CSVs here (not committed — see below)
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb         # full EDA + feature engineering + model comparison notebook
│   │   ├── predict.py                 # standalone script: load best model, predict one input
│   │   ├── figures/                   # saved plots
│   │   └── models/                    # best_model.pkl, comparison table
│   ├── API/
│   │   ├── prediction.py              # FastAPI app (predict, upload-data, retrain, retrain-status)
│   │   ├── requirements.txt
│   │   ├── models/best_model.pkl
│   │   └── data/incoming, data/processed   # background-watcher queue folders
│   └── FlutterApp/                    # Task 3 mobile app
└── pyproject.toml
```

## Getting the data (for retraining/exploration only — not needed to run the API)
Download and unzip into `data/`:
- https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india
- https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes

Expected files:
```
data/A_Z_medicines_dataset_of_India.csv
data/medicine_dataset.csv
```

## Running the notebook locally (uv)
```bash
uv sync
uv run jupyter notebook summative/linear_regression/multivariate.ipynb
```

## Model comparison summary

Four regression approaches were trained and compared (see the notebook for full detail):

| Model | Test RMSE (log-price) | Test R² |
|---|---|---|
| **Random Forest (best)** | 0.658 | 0.451 |
| Decision Tree | 0.682 | 0.409 |
| Linear Regression (OLS) | 0.752 | 0.283 |
| SGD Linear Regression (stochastic) | 0.757 | 0.273 |

**Random Forest** was saved as the best-performing model
(`summative/linear_regression/models/best_model.pkl`) — price is driven by
non-linear interactions between drug category and numeric attributes
(e.g. dosage strength matters differently within "Anti Neoplastics" than
within "Pain Analgesics"), a pattern only a tree-based model captures.

## API (Task 2)

Base URL: **[https://linear-regression-model-2-5ni5.onrender.com](https://linear-regression-model-2-5ni5.onrender.com)**
Swagger UI: **[https://linear-regression-model-2-5ni5.onrender.com](https://linear-regression-model-2-5ni5.onrender.com)/docs**

**Endpoints:**
- `GET /` — health check + last auto-retrain status summary
- `POST /predict` — predict a medicine's price from its 13 attributes (all Pydantic-validated with explicit datatypes and range/category constraints)
- `POST /upload-data` — queue a new labeled CSV; does **not** retrain by itself
- `GET /retrain-status` — check the background watcher: last check time, last retrain time, rows used, resulting RMSE, files queued
- `POST /retrain` — manual/instant retrain (admin fallback for quick testing)

**Automatic ("reactive") retraining:** a background thread polls
`data/incoming/` every 20 seconds. Any CSV dropped there via
`POST /upload-data` is picked up **on its own** — retrained, hot-swapped
into the running API, and archived — with no manual `/retrain` call
needed. See `summative/API/README.md` for full details and a demo
walkthrough.

**CORS:** configured with an explicit origin allow-list (not `*`),
restricted to `GET`/`POST` methods, explicit allowed headers, and
credentials disabled — reasoning is documented directly above the
`CORSMiddleware` call in `summative/API/prediction.py`.

## Mobile app (Task 3)

One page with 13 input fields (9 numeric text fields + 4 category
dropdowns, matching the model's 13 features), a **Predict** button, and a
result/error display area.

### Run instructions
1. Install the Flutter SDK: https://docs.flutter.dev/get-started/install
2. From `summative/FlutterApp/`:
   ```bash
   flutter pub get
   flutter devices        # confirm a device/emulator is available
   flutter run -d android # or your connected device/emulator id
   ```
3. The app is already pointed at the live API
   (`kApiBaseUrl` in `lib/main.dart` = `https://linear-regression-model-2-5ni5.onrender.com`) —
   no configuration needed to test it as-is.

See `summative/FlutterApp/README.md` for more detail.

## Video demo (Task 4)
https://youtu.be/kOFmQL-GrU4

Covers: mobile app making a live prediction, Swagger UI tests (including a
datatype/range validation failure), the Flutter code that calls the API,
a walkthrough of the notebook's model comparison, and the 4 required
questions (loss level & how to reduce it, hyperparameters, updating with
new data, CORS configuration reasoning).
