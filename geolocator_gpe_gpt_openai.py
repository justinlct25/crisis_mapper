import pandas as pd
import spacy
import config
import time
import random
from openai import OpenAI
from helper import get_latest_file
from datetime import datetime
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import os
from tqdm import tqdm 

# --- Setup ---
nlp = spacy.load("en_core_web_sm")
geolocator = Nominatim(user_agent="geoapiCrisis")
gpt_client = OpenAI(api_key=config.openai_key)
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
AVG_TOKENS_PER_POST = 100

# --- Batch Helpers ---
def batch_posts(posts, max_tokens=MAX_TOKENS, avg_tokens=AVG_TOKENS_PER_POST):
    batch, all_batches = [], []
    current_tokens = 0
    for post in posts:
        if current_tokens + avg_tokens > max_tokens:
            all_batches.append(batch)
            batch = [post]
            current_tokens = avg_tokens
        else:
            batch.append(post)
            current_tokens += avg_tokens
    if batch:
        all_batches.append(batch)
    return all_batches

def call_openai_batch(batch):
    prompt = """For each post, verify if the detected location is possibly where the author is living or encountered emotional distress.

Return \"Unknown\" if it's not or it is just an unrelated place or distant/past trip.

Format:
1. location or Unknown
2. location or Unknown
..."""

    system_message = """You verify whether a detected location from a post is where the author is likely living or encountered emotional distress

Accept if it seems the author:
- is living there or encountered emotional distress there,
- and it's not an unrelated place or a distant, clearly-ended trip that does not raise the emotional distress.
"""

    numbered_inputs = "\n".join([f"{i+1}. Post: {text} | Detected Location: {loc}" for i, (text, loc) in enumerate(batch)])

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt + "\n" + numbered_inputs}
    ]

    try:
        for attempt in range(2):
            response = gpt_client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
            )

            reply = response.choices[0].message.content.strip().split("\n")
            parsed = [line.split(". ", 1)[-1].strip() if ". " in line else "Unknown" for line in reply]

            if len(parsed) == len(batch):
                if attempt > 0:
                    print(f"✅ Reattempt {attempt} successful for batch of size {len(batch)}.")
                return parsed
            print(f"⚠️ Mismatch (attempt {attempt+1}): expected {len(batch)}, got {len(parsed)}")
            # print("Raw response:\n", reply)
        print(f"❌ Failed after 2 attempts for batch of size {len(batch)}.")
        return ["Unknown"] * len(batch)

    except Exception as e:
        print(f"GPT error: {e}")
        return ["Unknown"] * len(batch)

# --- Main Pipeline ---
def extract_gpe(text):
    doc = nlp(text)
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return gpes[0] if gpes else "Unknown"

def geocode_location(name):
    try:
        location = geolocator.geocode(name)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except:
        return pd.Series([None, None])
    return pd.Series([None, None])

def run_geolocation_pipeline(source='reddit', classified_posts_csv=None):
    latest_file, latest_time = get_latest_file('data/classified_posts', 'classified', classified_posts_csv)
    print(f"Loading latest classified posts file: {latest_file}")
    df = pd.read_csv(latest_file)

    # Step 1: Extract GPEs with spaCy
    print("Extracting locations with spaCy...")
    df["gpe"] = [
        extract_gpe(text) for text in tqdm(df["clean_content"], desc="Extracting GPEs with spaCy")
    ]

    # Step 2: Filter posts with detected GPEs
    posts_with_gpe = df[df["gpe"].str.lower() != "unknown"]
    print(f"✅ Detected {len(posts_with_gpe)} posts with GPEs out of {len(df)} total posts.")

    # Step 3: Validate GPEs with GPT
    print("Validating detected locations with GPT...")
    batches = batch_posts(posts_with_gpe[["clean_content", "gpe"]].values.tolist())
    validated_locations = []
    total_success_count = 0

    temp_output_path = f"data/geolocated_posts/temp/temp_gpe_gpt_locations_{latest_time}.csv"
    for i, batch in enumerate(batches):
        result = call_openai_batch(batch)
        batch_success_count = sum(1 for loc in result if loc and loc.lower() != "unknown")
        total_success_count += batch_success_count

        print(f"Batch {i + 1}/{len(batches)}: {batch_success_count}/{len(batch)} validated, "
              f"Total validated so far: {total_success_count}/{len(validated_locations) + len(batch)}")

        validated_locations.extend(result)
        time.sleep(random.uniform(1.2, 1.8))

        temp_df = posts_with_gpe.iloc[len(validated_locations) - len(result):len(validated_locations)].copy()
        temp_df["validated_location"] = result
        temp_df.to_csv(temp_output_path, mode='a', index=False, header=not os.path.exists(temp_output_path) if i == 0 else False)

    posts_with_gpe["validated_location"] = validated_locations
    posts_with_gpe = posts_with_gpe[
        posts_with_gpe["validated_location"].notna() & (posts_with_gpe["validated_location"].str.lower() != "unknown")
    ]

    # Step 4: Geocode Validated Locations
    print("Geocoding validated locations...")
    latitudes, longitudes = [], []
    for loc in tqdm(posts_with_gpe["validated_location"], desc="Geocoding locations"):
        time.sleep(random.uniform(1, 1.3))
        latlon = geocode_location(loc)
        latitudes.append(latlon[0])
        longitudes.append(latlon[1])

    posts_with_gpe["lat"], posts_with_gpe["lon"] = latitudes, longitudes
    posts_with_gpe = posts_with_gpe.dropna(subset=["lat", "lon"])

    # Step 5: Save Results
    output_path = f"data/geolocated_posts/geolocated_{len(posts_with_gpe)}_posts_{source}_by_gpe_gpt_{latest_time}.csv"
    posts_with_gpe.to_csv(output_path, index=False)
    print(f"Saved geolocated data to: {output_path}")

    # Step 6: Generate Heatmap
    print("Generating heatmap...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)
    cluster = MarkerCluster()

    for _, row in posts_with_gpe.iterrows():
        color = 'yellow'
        if str(row.get('risk_level_semantic', '')).startswith('High'):
            color = 'red'
        elif str(row.get('risk_level_semantic', '')).startswith('Moderate'):
            color = 'orange'

        popup = f"""
        <b>Location:</b> {row['validated_location']}<br>
        <b>Risk Level:</b> {row.get('risk_level_semantic', 'N/A')}<br>
        <b>Date:</b> {row.get('timestamp', 'N/A')}<br>
        <b>URL:</b> <a href="{row.get('url', '#')}" target="_blank">{row.get('url', '#')}</a>
        """
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(cluster)

    cluster.add_to(map_)
    heatmap_file = f"outputs/crisis_heatmap_{len(posts_with_gpe)}_{source}_by_gpe_gpt_filtered_{latest_time}.html"
    map_.save(heatmap_file)
    print(f"Heatmap saved to: {heatmap_file}")

# --- Execution ---
if __name__ == "__main__":
    run_geolocation_pipeline()
