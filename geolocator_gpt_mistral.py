import pandas as pd
import config
from geopy.geocoders import Nominatim
import time
import random
from helper import get_latest_file
from datetime import datetime
import folium
from folium.plugins import MarkerCluster
import requests

# Set up geocoder
geolocator = Nominatim(user_agent="geoapiCrisis")

# --- GPT-only location inference using Ollama + Mistral ---
def infer_location_with_gpt(text):
    prompt = f"""You are a strict information extractor.

        Your task is to extract the most likely **real-world location** (town, city, state, or country) mentioned or implied in the following post.

        ⚠️ Return the location *only* — no reasoning, no explanation, no extra words.
        ⚠️ If no location can be guessed, just return: Unknown

        Post: "{text}"
        Location:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )
        location = response.json()["response"].strip().split("\n")[0]
        print(f"Mistral model guessed location: {location}")
        return None if location.lower() in ["unknown", "none"] else location
    except Exception as e:
        print(f"Ollama Mistral model error: {e}")
        return None

def geolocate_posts_with_gpt(source='reddit', classified_posts_csv=None):
    # Load the latest classified posts file
    latest_file, latest_time_formatted = get_latest_file('data/classified_posts', 'classified', classified_posts_csv)
    print(f"Loading latest classified posts file: {latest_file}")
    df = pd.read_csv(latest_file)

    # --- Step 1: Use GPT to infer all locations ---
    print("Inferring locations with GPT only...")
    df['detected_location'] = df['clean_content'].apply(infer_location_with_gpt)

    # Drop rows with no detected location
    print("Filtering rows with valid locations...")
    df = df.dropna(subset=['detected_location'])

    # --- Step 2: Geocode all detected locations ---
    def geocode_location(location_name):
        try:
            location = geolocator.geocode(location_name)
            if location:
                return pd.Series([location.latitude, location.longitude])
        except:
            return pd.Series([None, None])
        return pd.Series([None, None])

    latitudes = []
    longitudes = []
    total = len(df)

    print("Geocoding detected locations...")
    for idx, loc in enumerate(df['detected_location']):
        time.sleep(random.uniform(1, 1.3))
        latlon = geocode_location(loc)
        latitudes.append(latlon[0])
        longitudes.append(latlon[1])
        print(f"Geocoding progress: {idx + 1}/{total}", end='\r')

    df['lat'] = latitudes
    df['lon'] = longitudes

    df = df.dropna(subset=['lat', 'lon'])

    output_file = f"data/geolocated_posts/geolocated_{len(df)}_posts_{source}_by_gpt_{latest_time_formatted}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved geolocated data to: {output_file}")

    # --- Step 3: Generate heatmap ---
    print("Generating heatmap with clustering...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)
    marker_cluster = MarkerCluster()

    for _, row in df.iterrows():
        if row['risk_level_semantic'].startswith('High'):
            color = 'red'
        elif row['risk_level_semantic'].startswith('Moderate'):
            color = 'orange'
        else:
            color = 'yellow'

        popup_content = f"""
        <b>Location:</b> {row['detected_location']}<br>
        <b>Risk Level:</b> {row['risk_level_semantic']}<br>
        <b>Date:</b> {row['timestamp']}<br>
        <b>URL:</b> <a href="{row['url']}" target="_blank">{row['url']}</a>
        """
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(marker_cluster)

    marker_cluster.add_to(map_)

    heatmap_file = f"outputs/crisis_heatmap_{len(df)}_{source}_posts_by_gpt_{latest_time_formatted}.html"
    map_.save(heatmap_file)
    print(f"\nHeatmap saved to: {heatmap_file}")

geolocate_posts_with_gpt()
