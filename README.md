# EV Charging Demand Forecasting Pipeline

A machine learning pipeline that predicts EV charging station utilization in Wisconsin using station characteristics, weather data, and temporal patterns.

## Overview

This project fetches EV charging station data from the NLR API and weather data from OpenWeatherMap, then trains an XGBoost regression model to forecast hourly utilization rates. An interactive Streamlit dashboard allows users to explore predictions and understand model behavior through SHAP explanations.

## Features

- **Data Extraction**: Pulls EV charging station data from NLR and current weather from OpenWeatherMap
- **Data Transformation**: Cleans and enriches data with features like fast-charger ratio and total ports
- **Utilization Simulation**: Generates realistic utilization patterns based on network type, time-of-day, and weather
- **ML Model Training**: XGBoost regressor with one-hot encoding for categorical features
- **Interactive Dashboard**: Streamlit app with maps, demand curves, and SHAP-based explanations

## Project Structure

```
├── src/
│   ├── run_pipeline.py      # Orchestrates the full ETL + training pipeline
│   ├── extract_api.py       # Fetches data from NLR and OpenWeatherMap APIs
│   ├── transform_load.py    # Cleans data and loads into SQLite
│   ├── simulate_target.py   # Generates simulated utilization data
│   ├── train_model.py       # Trains and evaluates XGBoost model
│   └── app.py               # Streamlit dashboard
├── data/
│   └── ev_stations.db       # SQLite database with station and forecast data
├── models/
│   ├── xgboost_ev_model.pkl # Trained XGBoost model
│   └── encoder.pkl          # Fitted ColumnTransformer for encoding
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── requirements.txt
└── .env.example
```

## Installation

1. Clone the repository
2. Create a conda environment or virtual environment with Python 3.13+
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your API keys:
   ```
   NLR_KEY=your_nlr_api_key_here
   WEATHER_KEY=your_openweathermap_api_key
   ```

## Usage

### Run the Full Pipeline

```bash
python src/run_pipeline.py
```

This executes all four steps:
1. Extract EV stations and weather data from APIs
2. Transform and load data into SQLite
3. Simulate utilization target variable
4. Train and save the XGBoost model

### Launch the Dashboard

```bash
streamlit run src/app.py
```

The dashboard includes:
- **Overview Tab**: Map of stations with quick stats
- **Station Detail Tab**: Hourly demand curve with SHAP explanations
- **Network Insights Tab**: Average utilization comparison across networks

## Model Features

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0-23) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `is_weekend` | Binary weekend indicator |
| `total_ports` | Total charging ports at station |
| `is_fast_charging_hub` | Has DC fast chargers |
| `fast_charger_ratio` | Proportion of DC fast chargers |
| `temperature` | Current temperature (°F) |
| `humidity` | Current humidity (%) |
| `wind_speed` | Current wind speed (mph) |
| `network` | Charging network (categorical) |
| `weather_condition` | Weather condition (categorical) |

## API Dependencies

- [NLR Alternative Fuel Stations API](https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/)
- [OpenWeatherMap Current Weather API](https://openweathermap.org/current)

## Tech Stack

- **Data**: pandas, SQLite
- **ML**: XGBoost, scikit-learn, SHAP
- **Visualization**: Streamlit, pydeck, matplotlib, seaborn
