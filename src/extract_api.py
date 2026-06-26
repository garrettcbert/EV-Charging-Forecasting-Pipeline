import requests
import os
from dotenv import load_dotenv
import json
import time


load_dotenv()

def run():
    NRL_API = os.getenv("NRL_KEY")
    WEATHER_API = os.getenv("WEATHER_KEY")

    ## Extract EV charging station data from NREL API and save to JSON
    NRL_URL = f"https://developer.nlr.gov/api/alt-fuel-stations/v1.json"

    params = {
        "api_key": NRL_API,
        "fuel_type": "ELEC",
        "limit": 200,
        "state": "WI"
    }

    response = requests.get(NRL_URL, params=params, timeout = 15)
    response.raise_for_status()

    ev_data = response.json()
    stations = ev_data['fuel_stations']

    with open("ev_charging_stations.json", "w") as f:
        json.dump(ev_data, f, indent=4)

    print(f"Extracted {len(stations)} EV charging stations and saved to ev_charging_stations.json")

    def round_coord(coord, precision = 2):
        return round(coord, precision)

    seen = set()
    unique_coords = []
    for station in stations:
        lat = round_coord(station['latitude'])
        lon = round_coord(station['longitude'])
        if (lat, lon) not in seen:
            seen.add((lat, lon))
            unique_coords.append((lat, lon))

    print(f"Found {len(unique_coords)} unique coordinates after rounding.")

    ## Extract Weather data from OpenWeatherMap API and save to JSON
    WEATHER_URL = f"https://api.openweathermap.org/data/2.5/weather"

    weather_lookup = {}

    for lat, lon in unique_coords:
        params = {
            "appid": WEATHER_API,
            "units": "imperial",
            "lat": lat,
            "lon": lon
        }
        response = requests.get(WEATHER_URL, params=params)
        if response.status_code == 200:
            weather_lookup[(lat, lon)] = response.json()
        else:
            print(f"Error fetching weather for ({lat}, {lon}): {response.status_code} - {response.text}")
            weather_lookup[(lat, lon)] = None

        time.sleep(0.1)  # Sleep to respect API rate limits

    serializable = {f"{lat},{lon}": data for (lat, lon), data in weather_lookup.items()}
    with open("weather_lookup.json", "w") as f:
        json.dump(serializable, f, indent=4)

    print(f"Extracted weather data for {len(weather_lookup)} unique coordinates and saved to weather_lookup.json")


if __name__ == "__main__":
    run()

