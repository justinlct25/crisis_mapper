# geolocate_from_text.py
import pandas as pd
import spacy
from geopy.geocoders import Nominatim
import folium
import time
import random

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="geoapiCrisis")

# Load your classified posts
df = pd.read_csv('data/classified_posts.csv')

# Extract location mentions from text
def extract_location(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations[0] if locations else None

# Apply location extraction
df['detected_location'] = df['clean_content'].apply(extract_location)

# Drop rows with no detected location
df = df.dropna(subset=['detected_location'])

# Geocode to get lat/lon
def geocode_location(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except:
        return pd.Series([None, None])
    return pd.Series([None, None])

# Use a progress-safe loop with pause to avoid rate-limiting
latitudes = []
longitudes = []

total_locations = len(df['detected_location'])
for idx, loc in enumerate(df['detected_location']):
    time.sleep(random.uniform(1, 1.3))  # polite delay
    latlon = geocode_location(loc)
    latitudes.append(latlon[0])
    longitudes.append(latlon[1])
    print(f"Geocoding progress: {idx + 1}/{total_locations}", end='\r')  # Inline progress

df['lat'] = latitudes
df['lon'] = longitudes

# Drop if geocoding failed
df = df.dropna(subset=['lat', 'lon'])

# Save geocoded data
df.to_csv('data/geolocated_posts.csv', index=False)

# 🔥 Plotting heatmap
map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['lat'], row['lon']),
        radius=6,
        color='red' if row['risk_level'] == 'High' else 'orange',
        fill=True,
        popup=f"{row['detected_location']} ({row['risk_level']})"
    ).add_to(map_)

map_.save('outputs/crisis_heatmap_from_text.html')
print("\nHeatmap saved to outputs/crisis_heatmap_from_text.html")