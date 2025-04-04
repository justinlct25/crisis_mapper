import pandas as pd
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helper import get_latest_file
from tqdm import tqdm
import torch
from datetime import datetime
import sys

def vader_avg_sentiment(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(sent)['compound'] for sent in sentences]
    return sum(scores) / len(scores)


def classify_posts_with_bert(source='reddit', extracted_posts_csv=None):
    # Load the latest extracted posts file
    latest_extracted_file, latest_time_formatted = get_latest_file('data/extracted_posts', 'extracted', extracted_posts_csv)
    print(f"Loading latest extracted posts file: {latest_extracted_file}")
    extracted_df = pd.read_csv(latest_extracted_file, comment='#')

    # Load the latest classified posts file
    try:
        latest_classified_file, _ = get_latest_file('data/classified_posts', 'classified')
        print(f"Loading latest classified posts file: {latest_classified_file}")
        classified_df = pd.read_csv(latest_classified_file, comment='#')
    except FileNotFoundError:
        print("No previously classified posts found. Starting fresh.")
        classified_df = pd.DataFrame(columns=extracted_df.columns)

    # Filter out already classified posts
    already_classified_ids = set(classified_df['id'])
    new_posts_df = extracted_df[~extracted_df['id'].isin(already_classified_ids)]
    print(f"Found {len(new_posts_df)} new posts to classify.")

    if new_posts_df.empty:
        print("No new posts to classify. Exiting.")
        return

    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define example phrases for each risk level
    risk_examples = {
        "High": [
            "I want to kill myself",
            "I don’t want to be here anymore",
            "I am planning to end it all",
            "I wish I was dead",
            "I can't live like this anymore"
        ],
        "Moderate": [
            "I feel lost",
            "I need help",
            "I'm overwhelmed by everything",
            "I can't handle this alone",
            "I'm struggling mentally"
        ],
        "Low": [
            "Mental health matters",
            "Everyone has tough days",
            "Take care of yourself",
            "I believe in self-care",
            "Talking helps"
        ]
    }

    # Flatten examples and store labels
    example_texts = []
    example_labels = []

    for label, examples in risk_examples.items():
        example_texts.extend(examples)
        example_labels.extend([label] * len(examples))

    # Encode example embeddings with progress printing
    example_embeddings = []
    for text in tqdm(example_texts, desc="Encoding examples"):
        example_embeddings.append(model.encode(text, convert_to_tensor=True))
    example_embeddings = torch.stack(example_embeddings)

    # Classify each post by similarity to risk examples
    post_texts = new_posts_df['clean_content'].tolist()
    post_embeddings = []
    for text in tqdm(post_texts, desc="Encoding posts"):
        post_embeddings.append(model.encode(text, convert_to_tensor=True))
    post_embeddings = torch.stack(post_embeddings)

    predicted_labels = []
    similarity_scores = []

    print("Computing similarities and classifying risk...")
    for post_emb in tqdm(post_embeddings, desc="Classifying posts"):
        scores = util.cos_sim(post_emb, example_embeddings)[0]
        best_idx = scores.argmax().item()
        predicted_labels.append(example_labels[best_idx])
        similarity_scores.append(float(scores[best_idx]))

    # Combine predicted labels and rounded similarity scores into a single column
    new_posts_df['risk_level_semantic'] = [
        f"{label} ({round(score, 2)})" for label, score in zip(predicted_labels, similarity_scores)
    ]

    # Add VADER sentiment scores
    print("Computing VADER sentiment scores...")
    new_posts_df['sentiment'] = [
        vader_avg_sentiment(text) for text in tqdm(new_posts_df['clean_content'], desc="Computing VADER sentiment")
    ]

    # Combine newly classified posts with already classified posts
    combined_df = pd.concat([classified_df, new_posts_df], ignore_index=True)

    # Reorder columns
    columns = ['id', 'timestamp', 'sentiment', 'risk_level_semantic'] + \
              [col for col in combined_df.columns if col not in ['id', 'timestamp', 'sentiment', 'risk_level_semantic']]
    combined_df = combined_df[columns]

    # Save result with timestamp
    output_file = f"data/classified_posts/classified_{len(combined_df)}_{source}_posts_by_semantic_{latest_time_formatted}.csv"
    with open(output_file, 'w') as f:
        f.write(f"# Extracted posts file: {latest_extracted_file}\n")
        combined_df.to_csv(f, index=False)
    print(f"Saved: {output_file}")

# Check command-line arguments
# if len(sys.argv) != 2 or sys.argv[1] not in ['r', 'x']:
#     print("Usage: python classifier_bert.py [r|x]")
#     print("r: Load latest extracted data from Reddit")
#     print("x: Load latest extracted data from X.com")
#     sys.exit(1)

# source = sys.argv[1]
# source = "reddit" if source == 'r' else "x"

if __name__ == '__main__':
    classify_posts_with_bert()
    # classify_posts_with_bert(extracted_posts_csv='data/extracted_posts/extracted_646_reddit_posts_by_keywords_20250327_234238.csv')