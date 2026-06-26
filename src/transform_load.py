import pandas as pd
import json
import sqlite3

def round_coord(coord, precision = 2):
    return round(coord, precision)

def run():
    ## Cleaing EV charging station
    with open("ev_charging_stations.json", "r") as f:
        data = json.load(f)

    cleaned_data = []
    for station in data['fuel_stations']:
        cleaned_data.append({
            "station_id": station['id'],
            "longitude": station['longitude'],
            "latitude": station['latitude'],
            "zip_code": station['zip'],
            "network": station['ev_network'] if station['ev_network'] else "Non-Networked",
            "dc_fast_ports": station['ev_dc_fast_num'] if station['ev_dc_fast_num'] else 0,
            "level2_ports": station['ev_level2_evse_num'] if station['ev_level2_evse_num'] else 0
        })

    stations_df = pd.DataFrame(cleaned_data)
    stations_df['dc_fast_ports'] = stations_df['dc_fast_ports'].astype(int)
    stations_df['level2_ports'] = stations_df['level2_ports'].astype(int)

    # Filter out stations with no ports
    stations_df = stations_df[stations_df['dc_fast_ports'] + stations_df['level2_ports'] > 0]

    # Create additional features
    stations_df['total_ports'] = stations_df['dc_fast_ports'] + stations_df['level2_ports']

    stations_df['is_fast_charging_hub'] = (stations_df['dc_fast_ports'] > 0).astype(int)

    stations_df['fast_charger_ratio'] = (stations_df['dc_fast_ports'] / stations_df['total_ports']).round(3)

    print(f'Stations after cleaning: {len(stations_df)}')
    print(stations_df['network'].value_counts())

    ## Cleaing Weather data
    with open("weather_lookup.json", "r") as f:
        weather_raw = json.load(f)

    cleaned_weather = []
    for key, weather_data in weather_raw.items():
        if weather_data is None:
            continue
        lat, lon = key.split(",")
        cleaned_weather.append({
            "latitude": float(lat),
            "longitude": float(lon),
            "city": weather_data['name'],
            "temperature": weather_data['main']['temp'],
            "humidity": weather_data['main']['humidity'],
            "wind_speed": weather_data['wind']['speed'],
            "weather_condition": weather_data['weather'][0]['main']
        })

    weather_df = pd.DataFrame(cleaned_weather)

    stations_df['rounded_lat'] = stations_df['latitude'].apply(round_coord)
    stations_df['rounded_lon'] = stations_df['longitude'].apply(round_coord)
    weather_df['rounded_lat'] = weather_df['latitude'].apply(round_coord)
    weather_df['rounded_lon'] = weather_df['longitude'].apply(round_coord)

    weather_df = weather_df.drop(columns=['latitude', 'longitude'])

    final_df = pd.merge(stations_df, weather_df, on=['rounded_lat', 'rounded_lon'], how='left')
    final_df.drop(columns=['rounded_lat', 'rounded_lon'], inplace=True)

    print(f'Final dataset: {len(final_df)} rows, {final_df.shape[1]} columns')
    print(final_df[['station_id', 'network', 'total_ports', 'temperature', 'weather_condition']].head())

    conn = sqlite3.connect('data/ev_stations.db')
    final_df.to_sql('station_snapshots', conn, if_exists='replace', index=False)
    conn.close()

    print("Data transformed and loaded into SQLite database successfully.")

if __name__ == "__main__":
    run()