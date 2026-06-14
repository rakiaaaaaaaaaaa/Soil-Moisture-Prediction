# 🌱 Soil Moisture Prediction — ML Ensemble Pipeline

Predicting soil moisture from satellite imagery, radar backscatter, and meteorological data using a gradient boosting ensemble.

---

## 📌 Overview

This project builds a regression pipeline to estimate **soil moisture** from multi-source geospatial inputs, including Sentinel-2 optical bands, SAR radar signals, weather time series, and soil texture properties.

It was developed for a Kaggle-style competition and achieves strong predictive performance through extensive feature engineering and a weighted ensemble of three boosting models.

---

## 📂 Data Sources

| Source | Description |
|---|---|
| Sentinel-2 | Optical satellite bands (B2, B3, B4, B5, B8, B8A, B11, B12) |
| SAR Radar | VV/VH backscatter + incidence angle |
| Meteorology | Hourly & daily temperature, precipitation, evaporation, pressure |
| Soil Texture | Sand, silt, clay percentages |
| Other | Elevation, cloud cover, climate zone, datetime |

---

## ⚙️ Pipeline

1. **Temporal Feature Extraction** — parses time-series columns into statistical summaries (mean, std, min, max, last, range)
2. **Feature Engineering** — creates 100+ features:
   - Spectral indices: NDVI, NDWI, NDMI, EVI, SAVI, BSI
   - Radar features: VV/VH ratio, RVI, polarization ratio, angle-normalized backscatter
   - Soil physics: water holding capacity, porosity, texture ratios
   - Meteorological: vapor pressure deficit (VPD), water balance (P − ET)
   - Cyclical time encoding (sin/cos of month and day-of-year)
   - Cross-feature interactions (e.g. clay × water balance, NDVI × precipitation)
3. **Encoding** — K-fold target encoding for climate zone (leakage-free)
4. **Modeling** — LightGBM, XGBoost, and CatBoost trained with early stopping
5. **Ensemble** — optimal weighted average found via grid search on validation R²
6. **Submission** — predictions exported to `submission_file.csv`

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- LightGBM, XGBoost, CatBoost

---


## 📊 Evaluation

Models are evaluated on **R²** and **RMSE** on an 80/20 train/validation split. The ensemble consistently outperforms individual models.
