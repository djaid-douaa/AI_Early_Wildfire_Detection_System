# AI Early Wildfire Detection System

🏆 **1st place, Huawei AI Hackathon (March 2026)**

A two-layer system that detects forest fires by fusing three data sources, then predicts how fast and how far a detected fire will spread over the next two hours. Alerts are graded LOW / MEDIUM / HIGH / CRITICAL.

## How it works

| Stage | What it does |
|---|---|
| **Data fusion** | Combines IoT-style sensor readings (temperature, humidity, wind, rain), weather / Fire Weather Index codes (FFMC, DMC, DC, ISI) and NASA FIRMS satellite detections (brightness, fire radiative power, confidence) |
| **Confidence score** | Rule-based 0–100 score computed before the ML model: satellite 35 pts, sensors 30 pts, weather/FWI 35 pts |
| **Layer 1: detection** | Gradient Boosting classifier on 16 features (sensors + weather + satellite + confidence score) |
| **Layer 2: spread** | Gradient Boosting regressors trained on 202 real fire events; predict burned area (ha) and spread speed (m/min), with Rothermel-inspired speed labels |
| **Alert engine** | Fuses confidence and fire probability into one score and maps it to an alert level and response actions |

## Data

- **UCI Forest Fires**: Cortez & Morais (2007), 517 records from Montesinho Natural Park, Portugal
- **NASA FIRMS**: active-fire satellite detections

Training uses UCI + FIRMS fused (837 samples). Evaluation uses a held-out UCI blind set (130 samples) with real labels. Satellite statistics are computed from the FIRMS training split only.

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib joblib
jupyter notebook Early_Wildfire_Detection.ipynb
```

Section 10 of the notebook loads the saved models (`*.pkl`) and runs inference on new readings without retraining.

## Limitations

- The dataset is small (517 UCI records), and satellite features are fused from a separate source rather than observed at the same time and place, so the scores are optimistic. Real deployment would need co-located sensor and satellite data.
- Spread speed labels are physics-inspired estimates, not measured values.

## Files

| File | Purpose |
|---|---|
| `Early_Wildfire_Detection.ipynb` | Full pipeline: data, fusion, models, alerts, plots |
| `detection_model.pkl`, `detection_scaler.pkl` | Fire classifier and its scaler |
| `area_model.pkl`, `speed_model.pkl`, `spread_scaler.pkl` | Spread regressors and their scaler |
| `forestfires.csv`, `nasa_firms.csv` | Input data |
