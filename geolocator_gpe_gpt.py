# geolocate_with_fallback.py
import pandas as pd
import spacy
import config
from geopy.geocoders import Nominatim
import time
import random
import openai  # Requires openai package
from helper import get_latest_file
from datetime import datetime
import sys
import folium
from folium.plugins import MarkerCluster

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Set up geocoder and OpenAI
geolocator = Nominatim(user_agent="geoapiCrisis")
openai.api_key = config.openai_key

def extract_location_GPE(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0] if locations else None

def infer_location_with_gpt(text):
    prompt = f"""
    Extract the most likely town, city, state, or country mentioned or implied in the following post. 
    Return only the name of a real city, state, or country that can be found on Google Maps. Be specific. 
    If there is no place clearly implied, return “Unknown”.    

    Post: "{text}"
    Location:
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        location = response['choices'][0]['message']['content'].strip()
        return None if location.lower() in ["unknown", "none"] else location
    except:
        return None

def geolocate_posts_with_gpt(source='reddit', classified_posts_csv=None):
    # Load the latest classified posts file
    latest_file, latest_time_formatted = get_latest_file('data/classified_posts', 'classified', classified_posts_csv)
    print(f"Loading latest classified posts file: {latest_file}")
    df = pd.read_csv(latest_file)

    # --- Step 1: Try extracting explicit location via spaCy ---
    print("Extracting locations with spaCy...")
    df['detected_location'] = df['clean_content'].apply(extract_location_GPE)

    # --- Step 2: Use GPT to guess implied location when spaCy fails ---
    # print("Inferring fallback locations with GPT...")
    # df['detected_location'] = df['detected_location'].fillna(df['clean_content'].apply(infer_location_with_gpt))
    # df['detected_location'] = df['clean_content'].apply(infer_location_with_gpt)

    # Drop rows with no detected location
    print("Filtering rows with valid locations...")
    df = df.dropna(subset=['detected_location'])

    # --- Step 3: Geocode all detected locations ---
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
        time.sleep(random.uniform(1, 1.3))  # polite rate limit
        latlon = geocode_location(loc)
        latitudes.append(latlon[0])
        longitudes.append(latlon[1])
        print(f"Geocoding progress: {idx + 1}/{total}", end='\r')

    df['lat'] = latitudes
    df['lon'] = longitudes

    # Drop rows where geocoding failed
    df = df.dropna(subset=['lat', 'lon'])

    # Save final output with a timestamped filename
    output_file = f"data/geolocated_posts/geolocated_{len(df)}_posts_{source}_by_semantic_{latest_time_formatted}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved geolocated data with fallback to: {output_file}")

    # --- Step 4: Generate a heatmap with clustering ---
    print("Generating heatmap with clustering...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)  # Centered on the US
    marker_cluster = MarkerCluster()

    # Add individual markers to the cluster
    for _, row in df.iterrows():
        # Determine marker color based on risk level
        if row['risk_level_semantic'].startswith('High'):
            color = 'red'
        elif row['risk_level_semantic'].startswith('Moderate'):
            color = 'orange'
        else:  # Low risk
            color = 'yellow'

        # Format popup with location, risk level, URL, and date
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

    # Add the cluster to the map
    marker_cluster.add_to(map_)

    heatmap_file = f"outputs/crisis_heatmap_{len(df)}_{source}_by_semantic_{latest_time_formatted}.html"
    map_.save(heatmap_file)
    print(f"\nHeatmap saved to: {heatmap_file}")


# Check command-line arguments
# if len(sys.argv) != 2 or sys.argv[1] not in ['r', 'x']:
#     print("Usage: python geolocator_gpt.py [r|x]")
#     print("r: Load latest extracted data from Reddit")
#     print("x: Load latest extracted data from X.com")
#     sys.exit(1)

# source = sys.argv[1]
# source = "reddit" if source == 'r' else "x"

geolocate_posts_with_gpt()