# Import the necessary modules
from extractor_reddit_keywords import extract_reddit_posts_keywords
from classifier_semantic import classify_posts_with_bert
from geolocator_ner_extract_gpt_validate import run_geolocation_pipeline

def run_pipeline():
    """
    Run the full pipeline: Extract -> Classify -> Geolocate.
    """
    try:
        print("Step 1: Extracting Reddit posts...")
        extract_reddit_posts_keywords()
        print("✅ Extraction completed.")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return

    try:
        print("\nStep 2: Classifying posts with BERT...")
        classify_posts_with_bert()
        print("✅ Classification completed.")
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return

    try:
        print("\nStep 3: Geolocating posts...")
        run_geolocation_pipeline()
        print("✅ Geolocation completed.")
    except Exception as e:
        print(f"❌ Error during geolocation: {e}")
        return
    
    # classify_posts_with_bert(classified_posts_csv='asef')
    # run_geolocation_pipeline(classified_posts_csv="data/classified_posts/classified_23431_reddit_posts_by_semantic_20250407_004759.csv", geolocation_processed_posts_csv="fasdklfj")
    # run_geolocation_pipeline()

if __name__ == "__main__":
    run_pipeline()