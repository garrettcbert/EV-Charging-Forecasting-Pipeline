import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random

# Load stations from database
conn = sqlite3.connect("data/ev_stations.db")
df = pd.read_sql_query("SELECT * FROM station_snapshots", conn)
conn.close()

print(f'Loaded {len(df)} stations from database.')

# Higher tier = higher base usage
network_tiers = {
    "Tesla Destination": 0.75,
    "Chargeback Network": 0.65,
    "Blink Network": 0.5,
    "Electrify America": 0.6,
    "EVgo": 0.55,
    "ZEFNET": 0.2,
    "Non-Networked": 0.1
}

def get_network_tier(network):
    for key in network_tiers:
        if key.lower() in str(network).lower():
            return network_tiers[key]
    return 0.40

def hour_weight(hour, is_weekend):
    if is_weekend:
        # Weekend: midday plateau, no sharp peaks
        if 10 <= hour <= 16:
            return 0.85
        elif 8 <= hour < 10 or 16 < hour <= 19:
            return 0.65
        elif 7 <= hour < 8 or 19 < hour <= 21:
            return 0.45
        elif 0 <= hour < 6:
            return 0.1
        else:
            return 0.25
        
    else:
        # Weekday: morning and evening peaks
        if 17 <= hour < 19:
            return 0.95
        elif 7 <= hour < 8:
            return 0.75
        elif 9 <= hour < 16:
            return 0.6
        elif 20 <= hour < 22:
            return 0.45
        elif 0 <= hour < 6:
            return 0.08
        else:
            return 0.25
        
def weather_adjustment(temp, condition):
    adjustment = 0.0

    if temp < 20:
        adjustment += 0.12 # Cold weather increases usage due to battery efficiency loss and cabin heating
    elif temp < 35:
        adjustment += 0.07 # Mildly cold weather can also increase usage
    elif temp > 90:
        adjustment += 0.1 # Hot weather increases usage due to AC use and battery cooling

    condition = str(condition).lower()
    if "snow" in condition or "ice" in condition:
        adjustment += 0.08
    elif "rain" in condition or "storm" in condition:
        adjustment += 0.05
    elif "clear" in condition:
        adjustment -= 0.01 # Clear weather can slightly decrease usage

    return adjustment

records = []
base_time = datetime(2024, 6, 1, 0, 0, 0, 0) # start of June

for day_offset in range(7):
    for hour in range(24):
        snapshot_time = base_time + timedelta(days=day_offset, hours=hour)
        is_weekend = snapshot_time.weekday() >= 5

        for _, station in df.iterrows():
            network_base = get_network_tier(station['network'])
            time_weight = hour_weight(hour, is_weekend)
            weather_adj = weather_adjustment(
                station.get('temperature', 60),
                station.get('weather_condition', 'clear')
            )
            fast_hub_bonus = 0.08 if station.get('is_fast_charging_hub', 0) == 1 else 0.0
            noise = random.gauss(0, 0.05) # Add some randomness

            utilization = network_base * time_weight + weather_adj + fast_hub_bonus + noise
            if utilization < 0.02:
                utilization = random.uniform(0.01, 0.05)
            else:
                utilization = float(np.clip(utilization, 0.0, 1.0))

            records.append({
                "station_id": station['station_id'],
                "snapshot_time": snapshot_time.isoformat(),
                "hour": hour,
                "day_of_week": snapshot_time.weekday(),
                "is_weekend": is_weekend,
                "network": station['network'],
                "total_ports": station['total_ports'],
                "is_fast_charging_hub": station['is_fast_charging_hub'],
                "fast_charger_ratio": station['fast_charger_ratio'],
                "temperature": station.get('temperature', None),
                "humidity": station.get('humidity', None),
                "wind_speed": station.get('wind_speed', None),
                "weather_condition": station.get('weather_condition', 'clear'),
                "utilization_rate": utilization
            })

simulate_df = pd.DataFrame(records)

print(f"Generated {len(simulate_df)} simulated utilization records.")
print(simulate_df[['snapshot_time', 'network', 'total_ports', 'is_fast_charging_hub', 'temperature', 'weather_condition', 'utilization_rate']].head())
print(f"Utilization stats - mean: {simulate_df['utilization_rate'].mean():.3f}, min: {simulate_df['utilization_rate'].min():.3f}, max: {simulate_df['utilization_rate'].max():.3f}")

conn = sqlite3.connect("data/ev_stations.db")
simulate_df.to_sql("utilization_forecast", conn, if_exists="replace", index=False)
conn.close()

print("\nWritten to data/ev_forecasting.db in table 'utilization_forecast'.")