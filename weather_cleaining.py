import json
import pandas as pd
import sqlite3

def clean_weather_data(rawdata):
    with open(rawdata, "r") as f:
        data = json.load(f)

    cleaned_data = {
        "city": data['name'],
        "country": data['sys']['country'],
        "temperature": data['main']['temp'],
        "humidity": data['main']['humidity'],
        "weather_description": data['weather'][0]['description']
    }

    cleaned_df = pd.DataFrame([cleaned_data])
    return cleaned_df

sql = """CREATE TABLE IF NOT EXISTS weather_forecast (
    city TEXT,
    country TEXT,
    temperature REAL,
    humidity INTEGER,
    weather_description TEXT
)"""

sqlite3.connect("ev_forecasting.db").execute(sql)
cleaned_df = clean_weather_data("weather_forecast.json")
cleaned_df.to_sql("weather_forecast", sqlite3.connect("ev_forecasting.db"), if_exists="replace", index=False)