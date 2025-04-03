import pandas as pd
import config
import time
import random
import openai
from openai import OpenAI
from helper import get_latest_file
from datetime import datetime
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import os

# --- Setup ---
geolocator = Nominatim(user_agent="geoapiCrisis")
gpt_client = OpenAI(api_key=config.openai_key)
MODEL = "gpt-3.5-turbo"  # or "o3-mini"
MAX_TOKENS = 3500
AVG_TOKENS_PER_POST = 120

# --- GPT Batched Inference ---
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
    numbered_posts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(batch)])
    prompt = f"""Extract the most likely real-world location (town, city, state, or country) mentioned or implied in each of the following social media posts. 
    Respond with one location per post as a numbered list. If no location can be inferred, respond with "Unknown".

    {numbered_posts}
    """
    try:
        response = gpt_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        parsed = [line.split(". ", 1)[-1].strip() for line in raw.split("\n") if line]

        if len(parsed) != len(batch):
            print(f"⚠️ Mismatch: expected {len(batch)}, got {len(parsed)}")
            print("Raw response:\n", raw)
            if len(parsed) < len(batch):
                parsed += ["Unknown"] * (len(batch) - len(parsed))
            else:
                parsed = parsed[:len(batch)]

        return parsed
    except Exception as e:
        print(f"GPT error: {e}")
        return ["Unknown"] * len(batch)

def infer_locations_batched(texts, full_df, latest_time):
    batches = batch_posts(texts)
    all_locations = []
    output_path = f"data/geolocated_posts/temp/temp_gpt_locations_{latest_time}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_success_count = 0 

    for i, batch in enumerate(batches):
        # print(f"Inferring batch {i + 1}/{len(batches)}...")
        locations = call_openai_batch(batch)

        batch_success_count = sum(1 for loc in locations if loc and loc.lower() != "unknown")
        total_success_count += batch_success_count

        # print(f"Batch {i + 1} result: {locations}")
        # print(f"Total posts located so far: {total_success_count}/{len(all_locations) + len(batch)}")
        print(f"Batch {i + 1}/{len(batches)}: {batch_success_count}/{len(batch)} located, "
            f"Total located so far: {total_success_count}/{len(all_locations) + len(batch)}")
        all_locations.extend(locations)

        temp_df = full_df.iloc[len(all_locations) - len(locations):len(all_locations)].copy()
        temp_df["detected_location"] = locations
        temp_df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path) if i == 0 else False)

        time.sleep(1.2)

    return all_locations

# --- Main Geolocation Flow ---
def geolocate_posts_with_gpt(source='reddit', classified_posts_csv=None):
    latest_file, latest_time = get_latest_file('data/classified_posts', 'classified', classified_posts_csv)
    print(f"Loading file: {latest_file}")
    df = pd.read_csv(latest_file)

    print("Inferring locations with GPT...")
    clean_texts = df["clean_content"].fillna("").tolist()
    detected_locations = infer_locations_batched(clean_texts, df, latest_time)
    df["detected_location"] = detected_locations
    df = df[df["detected_location"].notna() & (df["detected_location"].str.lower() != "unknown")]

    print("Geocoding locations...")
    def geocode_location(name):
        try:
            location = geolocator.geocode(name)
            if location:
                return pd.Series([location.latitude, location.longitude])
        except:
            return pd.Series([None, None])
        return pd.Series([None, None])

    latitudes, longitudes = [], []
    for idx, loc in enumerate(df["detected_location"]):
        time.sleep(random.uniform(1, 1.3))
        latlon = geocode_location(loc)
        latitudes.append(latlon[0])
        longitudes.append(latlon[1])
        print(f"Geocoding: {idx+1}/{len(df)}", end="\r")

    df["lat"], df["lon"] = latitudes, longitudes
    df = df.dropna(subset=["lat", "lon"])

    output_file = f"data/geolocated_posts/geolocated_{len(df)}_posts_{source}_by_gpt_{latest_time}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved geolocated data to: {output_file}")

    print("Generating heatmap...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)
    cluster = MarkerCluster()

    for _, row in df.iterrows():
        color = 'yellow'
        if str(row['risk_level_semantic']).startswith('High'):
            color = 'red'
        elif str(row['risk_level_semantic']).startswith('Moderate'):
            color = 'orange'

        popup = f"""
        <b>Location:</b> {row['detected_location']}<br>
        <b>Risk Level:</b> {row['risk_level_semantic']}<br>
        <b>Date:</b> {row['timestamp']}<br>
        <b>URL:</b> <a href="{row['url']}" target="_blank">{row['url']}</a>
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
    heatmap_file = f"outputs/crisis_heatmap_{len(df)}_{source}_by_gpt_{latest_time}.html"
    map_.save(heatmap_file)
    print(f"Heatmap saved to: {heatmap_file}")

if __name__ == '__main__':
    geolocate_posts_with_gpt()
    # geolocate_posts_with_gpt(classified_posts_csv='data/classified_posts/classified_646_reddit_posts_by_semantic_None.csv')
