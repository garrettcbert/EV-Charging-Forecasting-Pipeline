import streamlit as st
import pickle
import sqlite3
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title = 'EV Charging Demand Forecast',
    layout = "wide",
    initial_sidebar_state='expanded'
)

@st.cache_resource
def load_model_and_encoder():
    with open("models/xgboost_ev_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_model_and_encoder()

@st.cache_data
def load_station_data():
    conn = sqlite3.connect("data/ev_stations.db")
    df = pd.read_sql_query("SELECT * FROM station_snapshots", conn)
    conn.close()
    return df

stations_df = load_station_data()

st.sidebar.header("Filter Stations")

network_filter = st.sidebar.multiselect(
    'Network',
    options = stations_df['network'].unique(),
    default = stations_df['network'].unique()
)

filtered_stations = stations_df[stations_df['network'].isin(network_filter)]

station_options = filtered_stations.apply(
    lambda row: f"{row['city']} - {row['network']} (ID: {row['station_id']})",
    axis = 1
)

selected_label = st.sidebar.selectbox("Station", station_options)

selected_station = filtered_stations[
    station_options == selected_label
].iloc[0]

day_type = st.sidebar.radio("Day Type", ["Weekday", "Weekend"])

is_weekend_flag = 1 if day_type == "Weekend" else 0

hours = list(range(24))

prediction_rows = []
for hour in hours:
    prediction_rows.append({
        "hour": hour,
        # weekend coded as Sat and weekday coded as Tue
        "day_of_week": 5 if is_weekend_flag else 1,
        "is_weekend": is_weekend_flag,
        "total_ports": selected_station['total_ports'],
        "is_fast_charging_hub": selected_station["is_fast_charging_hub"],
        "fast_charger_ratio": selected_station["fast_charger_ratio"],
        "temperature": selected_station["temperature"],
        "humidity": selected_station["humidity"],
        "wind_speed": selected_station["wind_speed"],
        "num_connector_types": selected_station["num_connector_types"],
        "is_free_charging": selected_station["is_free_charging"],
        "is_public": selected_station["is_public"],
        "nearby_station_count": selected_station["nearby_station_count"],
        "station_age_years": selected_station["station_age_years"],
        "network": selected_station["network"],
        "weather_condition": selected_station["weather_condition"],
    })

pred_df = pd.DataFrame(prediction_rows)

categorical_cols = ['network', 'weather_condition']
numeric_cols = ['hour', 'day_of_week', 'is_weekend', 'total_ports',
                'is_fast_charging_hub', 'fast_charger_ratio',
                'temperature', 'humidity', 'wind_speed',
                'num_connector_types', 'is_free_charging', 'is_public',
                'nearby_station_count', 'station_age_years']

X_pred = pred_df[numeric_cols + categorical_cols]
X_pred_encoded = encoder.transform(X_pred)

predictions = model.predict(X_pred_encoded)
pred_df['predicted_utilization'] = predictions

network_colors = {
    "Tesla Destination":  [220, 80, 80],
    "Tesla":               [230, 130, 130],
    "ChargePoint Network": [70, 130, 200],
    "Electrify America":   [80, 180, 120],
    "Blink Network":       [230, 160, 60],
    "EVgo":                [160, 100, 200],
    "ZEFNET":              [140, 140, 140],
    "Non-Networked":       [110, 110, 110],
}

map_df = filtered_stations.copy()


# ISSUE: ALL MARKERS LOOK THE SAME SIZE
def get_color_for_row(row):
    if row['station_id'] == selected_station['station_id']:
        return [250, 0, 0] # gold
    return network_colors.get(row['network'], [0, 0, 0])

def get_radius_for_row(row):
    if row['station_id'] == selected_station['station_id']:
        return 1200
    return 350

# Highlight the selected station with a more distinct marker
map_df['color'] = map_df.apply(get_color_for_row, axis = 1)
map_df["radius"] = map_df.apply(get_radius_for_row, axis = 1)

layer = pdk.Layer(
    "ScatterplotLayer",
    data = map_df,
    get_position = ['longitude', 'latitude'],
    get_color = "color",
    get_radius = "radius",
    stroked = True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=2,
    pickable = True
)

view_state = pdk.ViewState(
    latitude = filtered_stations['latitude'].mean(),
    longitude= filtered_stations['longitude'].mean(),
    zoom = 8
)

tooltip = {
    "html": "<b>{city}</b><br/>Network: {network}<br/>Total Ports: {total_ports}",
    "style": {"backgroundColor": "steelblue", "color": "white"}
}

st.title("EV Charging Demand Forecaster")
st.markdown(
    "Explore predicted charging station utilization across Wisconsin. "
    "Select a station to see its predicted demand curve throughout the day."
)

tab1, tab2, tab3 = st.tabs(["Overview", "Station Detail", "Network Insights"])

with tab1:
    st.subheader("Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Stations", len(filtered_stations))
    col2.metric("Networks Represented", filtered_stations['network'].nunique())
    col3.metric("Fast-Charging Hubs", int(filtered_stations['is_fast_charging_hub'].sum()))
    col4.metric("Avg Predicted Utilization",
                f"{pred_df['predicted_utilization'].mean():.0%}")
    
    st.write("Selected station ID:", selected_station["station_id"])

    st.subheader("Station Locations")
    st.pydeck_chart(pdk.Deck(
        layers = [layer],
        initial_view_state= view_state,
        tooltip= tooltip
    ))


with tab2:
    st.subheader("Predicted Demand Curve")


    st.line_chart(pred_df.set_index('hour')['predicted_utilization'])
    st.caption(
        "Curve shape is primarily driven by time of day; network type and "
        "weather shift the overall demand level rather than the shape of the curve."
    )

    st.subheader("Why this Prediction?")

    selected_hour = st.slider("Hour to explain", 0, 23, 17)

    explain_row = pred_df[pred_df['hour'] == selected_hour]
    X_explain = explain_row[numeric_cols + categorical_cols]
    X_explain_encoded = encoder.transform(X_explain)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain_encoded)

    def clean_feature_name(name):
        if name.startswith("cat__"):
            raw = name[len("cat__"):]
            for col in categorical_cols:
                prefix = f"{col}_"
                if raw.startswith(prefix):
                    label = col.replace("_", " ").title()
                    value = raw[len(prefix):]
                    return f"{label}: {value}"
            return raw.replace("_", " ").title()
        if name.startswith("remainder__"):
            return name[len("remainder__"):].replace("_", " ").title()
        return name.replace("_", " ").title()

    raw_feature_names = encoder.get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in raw_feature_names]
    X_explain_df = pd.DataFrame(X_explain_encoded, columns = feature_names)

    fig, ax = plt.subplots(figsize = (8, 6))
    shap.plots.bar(
        shap.Explanation(
            values = shap_values[0],
            base_values = explainer.expected_value,
            data = X_explain_df.iloc[0],
            feature_names = feature_names
        ),
        show = False
    )
    plt.tight_layout()

    st.pyplot(fig)

with tab3:
    st.subheader("Average Utilization by Network")

    networks = filtered_stations['network'].unique()
    network_avg_rows = []

    for network in networks:
        sample_station = filtered_stations[filtered_stations['network'] == network].iloc[0]

        hourly_pred = []
        for hour in range(24):
            row = pd.DataFrame([{
                "hour": hour,
                "day_of_week": 1,
                "is_weekend": 0,
                "total_ports": sample_station["total_ports"],
                "is_fast_charging_hub": sample_station["is_fast_charging_hub"],
                "fast_charger_ratio": sample_station["fast_charger_ratio"],
                "temperature": sample_station["temperature"],
                "humidity": sample_station["humidity"],
                "wind_speed": sample_station["wind_speed"],
                "num_connector_types": sample_station["num_connector_types"],
                "is_free_charging": sample_station["is_free_charging"],
                "is_public": sample_station["is_public"],
                "nearby_station_count": sample_station["nearby_station_count"],
                "station_age_years": sample_station["station_age_years"],
                "network": sample_station["network"],
                "weather_condition": sample_station["weather_condition"],
            }])

            encoded_row = encoder.transform(row)
            hourly_pred.append(model.predict(encoded_row)[0])
        
        network_avg_rows.append({
            'network': network,
            'avg_predicted_utilization': sum(hourly_pred) / len(hourly_pred)
        })

    network_df = pd.DataFrame(network_avg_rows).sort_values(
        'avg_predicted_utilization', ascending = False
    )

    st.bar_chart(network_df.set_index('network')['avg_predicted_utilization'])

st.markdown("---")
st.caption(
    "Built by Garrett Seibert . "
    # Add github and LinkedIn links
)