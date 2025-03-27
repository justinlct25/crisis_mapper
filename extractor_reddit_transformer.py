# extract_posts.py
import praw
import pandas as pd
from datetime import datetime
import re
from sentence_transformers import SentenceTransformer, util
import helper

# Setup Reddit API client
reddit = helper.bot_login()

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the query and encode it
semantic_queries = [
    "I want to kill myself",
    "I feel hopeless",
    "I can’t keep going",
    "I need help",
    "I feel empty",
    "I'm struggling with addiction",
    "I'm overwhelmed by life",
    "I don't want to live anymore",
    "Nothing matters to me",
    "I cry myself to sleep"
]
threshold = 0.4  # Adjust this threshold as needed


query_embedding = model.encode(semantic_queries, convert_to_tensor=True)

posts_data = []

# Example subreddit
total_posts = 1000  # Define the total number of posts to process
list_of_posts = []  # Collect posts for embedding

for idx, submission in enumerate(reddit.subreddit("mentalhealth").hot(limit=total_posts)):
    print(f"Extracting Reddit Posts: {idx + 1}/{total_posts}", end='\r')  # Inline progress
    post_content = submission.title + " " + submission.selftext
    list_of_posts.append(post_content)
    posts_data.append({
        'id': submission.id,
        'timestamp': datetime.utcfromtimestamp(submission.created_utc),
        'content': post_content,
        'score': submission.score,
        'num_comments': submission.num_comments
    })

# Compute embeddings for all posts
print("\nComputing embeddings for all posts...")
post_embeddings = model.encode(list_of_posts, convert_to_tensor=True)

# Compute similarities (correct shape: [posts, queries])
print("Computing similarities...")
similarities = util.cos_sim(post_embeddings, query_embedding)  # shape: [1000, 10]

# OR logic — keep post if it matches *any* query above threshold
matches = (similarities >= threshold).any(dim=1)

# Get the top-matching query and score for each post
top_scores, top_query_idxs = similarities.max(dim=1)  # shape: [1000]

# Filter posts with insight
filtered_posts = []
for i, match in enumerate(matches):
    print(f"Filtering Posts: {i + 1}/{total_posts}", end='\r')  # Inline progress
    if match:
        post = posts_data[i]
        post['matched_query'] = semantic_queries[top_query_idxs[i]]
        post['similarity_score'] = float(top_scores[i])
        filtered_posts.append(post)

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

df = pd.DataFrame(filtered_posts)
df['clean_content'] = df['content'].apply(clean_text)

# Add timestamp to the file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"data/extracted_posts_transformer_{timestamp}.csv"
df.to_csv(output_file, index=False)
print(f"Saved: {output_file}")
