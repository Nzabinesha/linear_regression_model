# Pharmacy Price Predictor — Flutter App

A single-page Flutter app that calls the deployed FastAPI prediction
endpoint (`../API/prediction.py`) and displays the predicted medicine price.

## Before running
Open `lib/main.dart` and set `kApiBaseUrl` to your deployed Render URL:
```dart
const String kApiBaseUrl = "https://your-service-name.onrender.com";
```

## Run instructions

1. Install the Flutter SDK: https://docs.flutter.dev/get-started/install
2. From this folder:
   ```bash
   flutter pub get
   flutter devices        # confirm a device/emulator/browser is available
   flutter run             # pick your target device when prompted
   ```
   - To run in Chrome (fastest way to demo): `flutter run -d chrome`
   - To run on an Android emulator: start the emulator first, then `flutter run -d android`
   - To run on a physical phone: enable USB debugging (Android) or
     Developer Mode (iOS), plug in the device, then `flutter run`

## What the app does
- One page with text fields for every numeric feature the model needs, and
  dropdowns (also form fields) for the 4 categorical features — 13 inputs
  total, matching the model's feature count.
- A **Predict** button that POSTs the form values as JSON to `/predict`.
- A display area below the button that shows either:
  - the predicted price, or
  - a clear error message if a field is missing/out of range (validated
    client-side before the request is even sent) or if the server itself
    returns an error.

## Notes
- Client-side validation mirrors the API's Pydantic constraints (same
  min/max ranges and the same categorical option lists), so a user gets
  immediate feedback without waiting on a network round-trip.
- If the API is unreachable (e.g. Render free-tier cold start), the app
  shows "Could not reach the API: ..." rather than crashing.
