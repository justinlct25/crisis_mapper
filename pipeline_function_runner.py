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
        # Extract posts from Reddit
        # extract_reddit_posts_keywords()
        print("✅ Extraction completed.")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return

    try:
        print("\nStep 2: Classifying posts with BERT...")
        # Classify posts using semantic analysis
        # classify_posts_with_bert()
        print("✅ Classification completed.")
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return

    # run_geolocation_pipeline(classified_posts_csv="data/classified_posts/classified_5688_reddit_posts_semantic_20250331_203311.csv", geolocation_processed_posts_csv="data/geolocated_posts/all_646_reddit_posts_by_ner_detect_gpt_validate_20250402_205551.csv")
    # run_geolocation_pipeline(classified_posts_csv="data/classified_posts/classified_3_reddit_posts_by_semantic_20250331_000001.csv", geolocation_processed_posts_csv="fasdklfj")
    # run_geolocation_pipeline(classified_posts_csv="data/classified_posts/classified_646_reddit_posts_by_semantic_20250331_000000.csv", geolocation_processed_posts_csv="fasdklfj")
    # run_geolocation_pipeline()

    # run_geolocation_pipeline()
    try:
        print("\nStep 3: Geolocating posts...")
        # Geolocate posts using NER and GPT
        run_geolocation_pipeline(classified_posts_csv="data/classified_posts/classified_5688_reddit_posts_semantic_20250331_203311.csv", geolocation_processed_posts_csv="fasdklfj")
        # run_geolocation_pipeline()
        print("✅ Geolocation completed.")
    except Exception as e:
        print(f"❌ Error during geolocation: {e}")
        return

if __name__ == "__main__":
    run_pipeline()