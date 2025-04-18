import pandas as pd
import spacy
import config
import time
import random
from openai import OpenAI
from helper import get_latest_file, save_dataframe_with_metadata, get_metadata_from_classified_file
from datetime import datetime
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import os
from tqdm import tqdm 
from pathlib import Path
import tiktoken
from generator_geolocated_crisis_heatmap import generate_heatmap

# --- Setup ---
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "lemmatizer"])
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

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in a given text for a specific model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def call_openai_batch(batch, max_context_tokens=16385, model="gpt-3.5-turbo"):
    """
    Process a batch of posts with GPT, ensuring the token count does not exceed the context limit.
    """
    prompt = """For each post, choose the **one most appropriate and specific** location from the detected list. If no valid location applies, return "Unknown".

Format:
1. normalized_location or Unknown
2. normalized_location or Unknown
...

Rules:
- Choose only one location per post — never return a list.
- Prefer specific and emotionally relevant locations (e.g., "Sheffield" over "United Kingdom").
- Use context to guide your decision (e.g., "I live in", "I feel", "I'm from").

Example:
Post: "I was born in France, lived in the US, but now I live in Austin, Texas." | Detected: ['France', 'US', 'Austin, Texas']  
Output: Austin, Texas
"""

    system_message = """You are verifying and normalizing the most relevant location from social media posts.

Instructions:
1. Your goal is to identify **one** location that best reflects where the author is currently living or where they are experiencing emotional distress.
2. If multiple locations are mentioned, choose the **most specific** and **emotionally relevant** one:
   - Prefer specific cities or regions (e.g., "Sheffield", "Niedersachsen") over general countries (e.g., "UK", "Germany").
   - Use cues like "I live in", "I feel", "I am from", or descriptions of daily life to determine the most personal location.
   - Ignore locations that refer to past places, online references, or unrelated trips unless no better option is available.

3. Normalize the chosen location:
   - Expand abbreviations (e.g., "UK" → "United Kingdom").
   - Use correct capitalization (e.g., "paris" → "Paris").

4. Return only the **final normalized location string** — never a list, combination, or multiple guesses.

If no location fits the above criteria, return "Unknown".
"""

    # Prepare the numbered inputs
    numbered_inputs = "\n".join([
        f"{i+1}. Post: {text} | Detected Locations: {gpes}" 
        for i, (text, gpes) in enumerate(batch)
    ])

    # Calculate token count for the full message
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt + "\n" + numbered_inputs}
    ]
    total_tokens = sum(count_tokens(message["content"], model=model) for message in messages)

    # If token count exceeds the limit, split the batch
    if total_tokens > max_context_tokens:
        print(f"⚠️ Token limit exceeded ({total_tokens} tokens). Splitting batch...")
        mid = len(batch) // 2
        return call_openai_batch(batch[:mid], max_context_tokens, model) + \
               call_openai_batch(batch[mid:], max_context_tokens, model)

    # Proceed with GPT API call
    try:
        for attempt in range(4):
            response = gpt_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            reply = response.choices[0].message.content.strip().split("\n")
            parsed = []
            for line in reply:
                line = line.strip()
                if not line:
                    parsed.append("Unknown")
                elif ". " in line:
                    parsed.append(line.split(". ", 1)[-1].strip())
                else:
                    parsed.append(line)
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

# --- Utility Functions ---
def extract_gpe(text):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return gpes

def geocode_location(name):
    try:
        location = geolocator.geocode(name)
        if location:
            return pd.Series([location.latitude, location.longitude])
    except:
        return pd.Series([None, None])
    return pd.Series([None, None])

# --- Main Pipeline ---
def run_geolocation_pipeline(source='reddit', classified_posts_csv=None, geolocation_processed_posts_csv=None):
    latest_classified_file, latest_classified_time = get_latest_file('data/classified_posts', 'classified', specified_file=classified_posts_csv)
    print(f"Loading latest classified posts file: {latest_classified_file}")
    classified_df = pd.read_csv(latest_classified_file, comment='#')

    classified_filename = Path(latest_classified_file).stem
    intermediate_dir = f"data/geolocated_posts/geolocation_process_{classified_filename}"
    os.makedirs(intermediate_dir, exist_ok=True)

    extracted_posts_file = get_metadata_from_classified_file(latest_classified_file, "Extracted posts file")
    metadata = {
        "Extracted posts file": extracted_posts_file,
        "Classified posts file": latest_classified_file
    }

    STEP1_OUTPUT_FILE = os.path.join(intermediate_dir, "step1_gpe_detected.csv")
    STEP3_OUTPUT_FILE = os.path.join(intermediate_dir, "step3_validated_locations.csv")
    STEP4_OUTPUT_FILE = os.path.join(intermediate_dir, "step4_geocoded_locations.csv")

    try:
        latest_all_posts_file, _ = get_latest_file('data/geolocated_posts', 'all', specified_file=geolocation_processed_posts_csv)
        print(f"Loading latest all_posts file: {latest_all_posts_file}")
        geo_processed_posts_df = pd.read_csv(latest_all_posts_file, comment='#')

        # Ensure required columns exist in geo_processed_posts_df
        required_columns = ["id", "detected_location", "validated_location", "lat", "lon"]
        for col in required_columns:
            if col not in geo_processed_posts_df.columns:
                geo_processed_posts_df[col] = None

        if len(classified_df) >= len(geo_processed_posts_df):
            geo_processed_posts_df = classified_df.merge(
                geo_processed_posts_df[required_columns],
                on="id",
                how="right",
                suffixes=("", "_geo")
            )
        else:
            geo_processed_posts_df = geo_processed_posts_df.merge(
                classified_df,
                on="id",
                how="left",
                suffixes=("_geo", "")
            )

        # Fill missing geolocation-related values with defaults
        geo_processed_posts_df["detected_location"] = geo_processed_posts_df["detected_location"].fillna("Unknown")
        geo_processed_posts_df["validated_location"] = geo_processed_posts_df["validated_location"].fillna("Unknown")
        geo_processed_posts_df["lat"] = geo_processed_posts_df["lat"].fillna(pd.NA)
        geo_processed_posts_df["lon"] = geo_processed_posts_df["lon"].fillna(pd.NA)

        # Retain only geolocated posts for already_geolocated_posts_df
        already_geolocated_posts_df = geo_processed_posts_df.dropna(subset=["lat", "lon"])
        print(f"Loaded {len(already_geolocated_posts_df)} geolocated posts from all_posts file.")
    except FileNotFoundError:
        print("No previously processed all_posts file found. Starting fresh.")
        geo_processed_posts_df = classified_df.copy()
        geo_processed_posts_df["detected_location"] = "Unknown"
        geo_processed_posts_df["validated_location"] = "Unknown"
        geo_processed_posts_df["lat"] = None
        geo_processed_posts_df["lon"] = None
        already_geolocated_posts_df = pd.DataFrame(columns=classified_df.columns)

    already_processed_ids = set(geo_processed_posts_df['id'])
    unprocessed_posts_df = classified_df[~classified_df['id'].isin(already_processed_ids)]
    print(f"Found {len(unprocessed_posts_df)} new posts to process.")

    if unprocessed_posts_df.empty:
        print("No new posts to process. Generating heatmap for existing geolocated posts.")
        already_geolocated_posts_df.to_csv(STEP1_OUTPUT_FILE, index=False)
        already_geolocated_posts_df.to_csv(STEP3_OUTPUT_FILE, index=False)
        already_geolocated_posts_df.to_csv(STEP4_OUTPUT_FILE, index=False)

        all_posts_df = geo_processed_posts_df.copy()
        all_posts_output_file = f"data/geolocated_posts/all_{len(all_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.csv"
        all_posts_output_copy = os.path.join(intermediate_dir, os.path.basename(all_posts_output_file))
        save_dataframe_with_metadata(all_posts_df, all_posts_output_file, metadata, copy_file=all_posts_output_copy)

        geolocated_only_posts_output_file = f"data/geolocated_posts/geolocated_{len(already_geolocated_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.csv"
        geolocated_only_posts_output_copy = os.path.join(intermediate_dir, os.path.basename(geolocated_only_posts_output_file))
        save_dataframe_with_metadata(already_geolocated_posts_df, geolocated_only_posts_output_file, metadata, copy_file=geolocated_only_posts_output_copy)

        heatmap_output_file = f"outputs/heatmap/crisis_heatmap_geolocated_{len(already_geolocated_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.html"
        heatmap_output_copy = os.path.join(intermediate_dir, os.path.basename(heatmap_output_file))
        generate_heatmap(already_geolocated_posts_df, output_file=heatmap_output_file)
        generate_heatmap(already_geolocated_posts_df, output_file=heatmap_output_copy)

        print(f"All outputs generated using existing geolocated posts.")
        return

    # --- Step 1: Extract GPEs ---
    if os.path.exists(STEP1_OUTPUT_FILE):
        gpe_detected_posts_df = pd.read_csv(STEP1_OUTPUT_FILE)
        gpe_detected_posts_df["detected_location"] = gpe_detected_posts_df["detected_location"].apply(eval)
    else:
        # Initialize detected_location as an empty list for all rows
        unprocessed_posts_df["detected_location"] = [[] for _ in range(len(unprocessed_posts_df))]

        # Extract GPEs for posts with valid content
        unprocessed_posts_df["detected_location"] = [
            extract_gpe(text) for text in tqdm(unprocessed_posts_df["clean_content"], desc="Extracting GPEs with spaCy")
        ]

        # Set detected_location to "None" for posts without any detected GPEs
        unprocessed_posts_df["detected_location"] = unprocessed_posts_df["detected_location"].apply(
            lambda x: x if isinstance(x, list) and len(x) > 0 else []
        )

        # Filter posts with detected GPEs for further processing
        gpe_detected_posts_df = unprocessed_posts_df[
            unprocessed_posts_df["detected_location"].apply(lambda x: isinstance(x, list) and len(x) > 0)
        ]
        gpe_detected_posts_df.to_csv(STEP1_OUTPUT_FILE, index=False)

    # --- Step 3: GPT Validation ---
    if os.path.exists(STEP3_OUTPUT_FILE):
        gpe_validated_posts_df = pd.read_csv(STEP3_OUTPUT_FILE)
    else:
        batches = batch_posts(gpe_detected_posts_df[["clean_content", "detected_location"]].values.tolist())
        validated_locations = []
        total_success_count = 0
        for i, batch in enumerate(batches):
            result = call_openai_batch(batch)
            batch_success_count = sum(1 for loc in result if loc and loc.lower() != "unknown")
            total_success_count += batch_success_count
            print(f"Batch {i + 1}/{len(batches)}: {batch_success_count}/{len(batch)} validated, Total validated so far: {total_success_count}/{len(validated_locations) + len(batch)}")
            validated_locations.extend(result)
            time.sleep(random.uniform(0.3, 0.8))
        gpe_validated_posts_df = gpe_detected_posts_df.copy()
        gpe_validated_posts_df["validated_location"] = validated_locations
        gpe_validated_posts_df.to_csv(STEP3_OUTPUT_FILE, index=False)

    # --- Step 4: Geocoding ---
    if os.path.exists(STEP4_OUTPUT_FILE):
        gpe_validated_posts_df = pd.read_csv(STEP4_OUTPUT_FILE)
    else:
        latitudes, longitudes = [], []
        for loc in tqdm(gpe_validated_posts_df["validated_location"], desc="Geocoding locations"):
            time.sleep(random.uniform(0.2, 0.6))
            latlon = geocode_location(loc)
            latitudes.append(latlon[0])
            longitudes.append(latlon[1])
        gpe_validated_posts_df["lat"], gpe_validated_posts_df["lon"] = latitudes, longitudes
        gpe_validated_posts_df = gpe_validated_posts_df.dropna(subset=["lat", "lon"])
        gpe_validated_posts_df.to_csv(STEP4_OUTPUT_FILE, index=False)

    # --- Save Results ---
    # Combine newly geolocated posts with previously geolocated posts
    combined_geolocated_posts_df = pd.concat([gpe_validated_posts_df, already_geolocated_posts_df], ignore_index=True)
    geolocated_only_posts_output_file = f"data/geolocated_posts/geolocated_{len(combined_geolocated_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.csv"
    geolocated_only_posts_output_copy = os.path.join(intermediate_dir, os.path.basename(geolocated_only_posts_output_file))
    save_dataframe_with_metadata(combined_geolocated_posts_df, geolocated_only_posts_output_file, metadata, copy_file=geolocated_only_posts_output_copy)

    # Update unprocessed posts
    unprocessed_posts_df = unprocessed_posts_df.copy()
    unprocessed_posts_df["detected_location"] = [[] for _ in range(len(unprocessed_posts_df))]
    unprocessed_posts_df["validated_location"] = "Unknown"
    unprocessed_posts_df["lat"] = None
    unprocessed_posts_df["lon"] = None
    gpe_validated_subset = gpe_validated_posts_df[["id", "detected_location", "validated_location", "lat", "lon"]]
    unprocessed_posts_df = unprocessed_posts_df.merge(gpe_validated_subset, on="id", how="left", suffixes=("", "_new"))
    for col in ["validated_location", "lat", "lon"]:
        unprocessed_posts_df[col] = unprocessed_posts_df[f"{col}_new"].combine_first(unprocessed_posts_df[col])
        unprocessed_posts_df.drop(columns=[f"{col}_new"], inplace=True)

    all_posts_df = pd.concat([geo_processed_posts_df, unprocessed_posts_df], ignore_index=True)
    all_posts_output_file = f"data/geolocated_posts/all_{len(all_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.csv"
    all_posts_output_copy = os.path.join(intermediate_dir, os.path.basename(all_posts_output_file))

    all_posts_output_file = f"data/geolocated_posts/all_{len(all_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.csv"
    all_posts_output_copy = os.path.join(intermediate_dir, os.path.basename(all_posts_output_file))
    save_dataframe_with_metadata(all_posts_df, all_posts_output_file, metadata, copy_file=all_posts_output_copy)

    heatmap_output_file = f"outputs/heatmap/crisis_heatmap_geolocated_{len(combined_geolocated_posts_df)}_{source}_posts_by_ner_detect_gpt_validate_{latest_classified_time}.html"
    heatmap_output_copy = os.path.join(intermediate_dir, os.path.basename(heatmap_output_file))
    generate_heatmap(geolocated_only_posts_output_file, output_file=heatmap_output_file)
    generate_heatmap(geolocated_only_posts_output_file, output_file=heatmap_output_copy)

# --- Execute ---
if __name__ == "__main__":
    run_geolocation_pipeline()
