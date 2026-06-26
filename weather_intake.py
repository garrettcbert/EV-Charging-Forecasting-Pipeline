import requests
import json

api_key = "1e7d5e2979387df78fa74d99127e30ce"
url = f"https://api.openweathermap.org/data/2.5/weather"

params = {
    "appid": api_key,
    "units": "imperial",
    "q": "Madison,US"
}

response = requests.get(url, params = params)
if response.status_code == 200:
    data = response.json()
    with open("weather_forecast.json", "w") as f:
        json.dump(data, f, indent=4)
else:
    print(f"Error: {response.status_code} - {response.text}")