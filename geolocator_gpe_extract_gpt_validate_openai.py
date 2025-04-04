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
    """
    Validate and normalize detected GPEs using GPT. For each post, GPT will decide which GPE is most appropriate,
    normalize its name, or return 'Unknown' if none are related.
    """
    prompt = """For each post, verify and normalize the detected locations. If none of the locations are related, return "Unknown".

Format:
1. normalized_location or Unknown
2. normalized_location or Unknown
..."""

    system_message = """You are tasked with verifying and normalizing detected locations from posts. 

For each post:
1. Verify whether a detected location is where the author is likely living or encountered emotional distress.
   - Accept a location if it seems the author is living there or encountered emotional distress there.
   - Reject a location if it is unrelated, such as a location mentioned in an online interaction, a distant/past trip, or clearly not relevant to the author's current or recent situation.

2. Normalize the location name:
   - Convert abbreviations to their full forms (e.g., "UK" → "United Kingdom", "USA" → "United States").
   - Ensure proper capitalization (e.g., "paris" → "Paris").

If none of the detected locations are appropriate, return "Unknown".

Your goal is to ensure that only relevant and meaningful locations are accepted and properly normalized."""

    # Format the batch for GPT
    numbered_inputs = "\n".join([
        f"{i+1}. Post: {text} | Detected Locations: {gpes}" 
        for i, (text, gpes) in enumerate(batch)
    ])

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
        print(f"❌ Failed after 2 attempts for batch of size {len(batch)}.")
        return ["Unknown"] * len(batch)

    except Exception as e:
        print(f"GPT error: {e}")
        return ["Unknown"] * len(batch)

# --- Main Pipeline ---
def extract_gpe(text):
    if not isinstance(text, str):
        return "Unknown"
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

def generate_heatmap(df, source, classified_file, geolocated_file):
    print("Generating heatmap...")
    map_ = folium.Map(location=[39.5, -98.35], zoom_start=4)
    cluster = MarkerCluster()

    for _, row in df.iterrows():
        color = 'yellow'
        if str(row.get('risk_level_semantic', '')).startswith('High'):
            color = 'red'
        elif str(row.get('risk_level_semantic', '')).startswith('Moderate'):
            color = 'orange'

        popup = f"""
        <b>Location:</b> {row['validated_location']}<br>
        <b>Sentiment:</b> {row.get('sentiment', 'N/A')}<br>
        <b>Risk Level:</b> {row.get('risk_level_semantic', 'N/A')}<br>
        <b>Date:</b> {row.get('timestamp', 'N/A')}<br>
        <b>URL:</b> <a href="{row.get('url', '#')}" target="_blank">{row.get('url', '#')}</a>
        """
        # <b>Classified File:</b> {classified_file}<br>
        # <b>Geolocated File:</b> {geolocated_file}<br>
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=300)
        ).add_to(cluster)

    cluster.add_to(map_)
    heatmap_file = f"outputs/crisis_heatmap_{len(df)}_{source}_by_ner_gpt_filtered.html"
    map_.save(heatmap_file)
    print(f"Heatmap saved to: {heatmap_file}")

def run_geolocation_pipeline(source='reddit', classified_posts_csv=None):
    # Load the latest classified posts file
    latest_classified_file, latest_classified_time = get_latest_file('data/classified_posts', 'classified', classified_posts_csv)
    print(f"Loading latest classified posts file: {latest_classified_file}")
    classified_df = pd.read_csv(latest_classified_file, comment='#')

    # Load the latest geolocated posts file
    try:
        latest_geolocated_file, _ = get_latest_file('data/geolocated_posts', 'geolocated')
        print(f"Loading latest geolocated posts file: {latest_geolocated_file}")
        geolocated_df = pd.read_csv(latest_geolocated_file, comment='#')
    except FileNotFoundError:
        print("No previously geolocated posts found. Starting fresh.")
        geolocated_df = pd.DataFrame(columns=classified_df.columns)

    # Filter out already geolocated posts
    already_geolocated_ids = set(geolocated_df['id'])
    new_posts_df = classified_df[~classified_df['id'].isin(already_geolocated_ids)]
    print(f"Found {len(new_posts_df)} new posts to geolocate.")

    if new_posts_df.empty:
        print("No new posts to geolocate. Generating heatmap for existing geolocated posts.")
        generate_heatmap(geolocated_df, source, latest_classified_file, latest_geolocated_file)
        return

    # Step 1: Extract GPEs by performing NER with spaCy
    new_posts_df = new_posts_df[new_posts_df["clean_content"].notna()]
    new_posts_df["clean_content"] = new_posts_df["clean_content"].astype(str)
    new_posts_df["detected_location"] = [
        extract_gpe(text) for text in tqdm(new_posts_df["clean_content"], desc="Extracting GPEs with spaCy")
    ]

    # Step 2: Filter posts with detected GPEs
    posts_with_gpe = new_posts_df[new_posts_df["detected_location"].apply(lambda x: x != "Unknown")]
    print(f"✅ Detected {len(posts_with_gpe)} posts with GPEs out of {len(new_posts_df)} new posts.")

    # Step 3: Validate and Normalize GPEs with GPT
    if posts_with_gpe.empty:
        print("No posts with detected GPEs to validate. Skipping GPT validation.")
        validated_locations = []
    else:
        print("Validating and normalizing detected locations with GPT...")
        batches = batch_posts(posts_with_gpe[["clean_content", "detected_location"]].values.tolist())
        validated_locations = []
        total_success_count = 0

        for i, batch in enumerate(batches):
            result = call_openai_batch(batch)
            batch_success_count = sum(1 for loc in result if loc and loc.lower() != "unknown")
            total_success_count += batch_success_count

            print(f"Batch {i + 1}/{len(batches)}: {batch_success_count}/{len(batch)} validated, "
                f"Total detected location validated so far: {total_success_count}/{len(validated_locations) + len(batch)}")

            validated_locations.extend(result)
            time.sleep(random.uniform(1.2, 1.8))

        posts_with_gpe["validated_location"] = validated_locations
        posts_with_gpe = posts_with_gpe[
            posts_with_gpe["validated_location"].notna() & (posts_with_gpe["validated_location"].str.lower() != "unknown")
        ]

    # Step 4: Geocode Validated Locations
    if not posts_with_gpe.empty:
        print("Geocoding validated locations...")
        latitudes, longitudes = [], []
        for loc in tqdm(posts_with_gpe["validated_location"], desc="Geocoding locations"):
            time.sleep(random.uniform(1, 1.3))
            latlon = geocode_location(loc)
            latitudes.append(latlon[0])
            longitudes.append(latlon[1])

        posts_with_gpe["lat"], posts_with_gpe["lon"] = latitudes, longitudes
        posts_with_gpe = posts_with_gpe.dropna(subset=["lat", "lon"])

    # Combine newly geolocated posts with previously geolocated posts
    combined_df = pd.concat([geolocated_df, posts_with_gpe], ignore_index=True)

    # Step 5: Save Results with Metadata (Geolocated Posts Only)
    output_file = f"data/geolocated_posts/geolocated_{len(combined_df)}_{source}_posts_by_ner_gpt_{latest_classified_time}.csv"
    with open(output_file, 'w') as f:
        f.write(f"# Extracted posts file: {classified_posts_csv}\n")
        f.write(f"# Classified posts file: {latest_classified_file}\n")
        combined_df.to_csv(f, index=False)
    print(f"Saved geolocated data to: {output_file}")

    # Step 6: Create a DataFrame for All Posts
    print("Creating a DataFrame for all posts...")
    all_posts_df = classified_df.copy()
    all_posts_df["detected_location"] = "Unknown"  # Default to "Unknown"
    all_posts_df["validated_location"] = "Unknown"  # Default to "Unknown"

    # Update detected and validated locations for posts with GPEs
    if not new_posts_df.empty:
        all_posts_df.loc[all_posts_df["id"].isin(new_posts_df["id"]), "detected_location"] = new_posts_df.set_index("id")["detected_location"]
    if not posts_with_gpe.empty:
        all_posts_df.loc[all_posts_df["id"].isin(posts_with_gpe["id"]), "validated_location"] = posts_with_gpe.set_index("id")["validated_location"]

    # Save All Posts with Metadata
    all_posts_output_file = f"data/geolocated_posts/all_posts_{len(all_posts_df)}_{source}_by_ner_gpt_{latest_classified_time}.csv"
    with open(all_posts_output_file, 'w') as f:
        f.write(f"# Extracted posts file: {classified_posts_csv}\n")
        f.write(f"# Classified posts file: {latest_classified_file}\n")
        f.write(f"# Geolocated posts file: {output_file}\n")
        all_posts_df.to_csv(f, index=False)
    print(f"Saved all posts data to: {all_posts_output_file}")

    # Step 7: Generate Heatmap
    generate_heatmap(combined_df, source, latest_classified_file, output_file)



# --- Execution ---
if __name__ == "__main__":
    run_geolocation_pipeline()
