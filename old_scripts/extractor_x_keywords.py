# extract_posts.py
import pandas as pd
from datetime import datetime
import re
import snscrape.modules.twitter as sntwitter
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Keywords to search
keywords = [
    "depressed", "anxiety", "panic attack", "suicidal", "overdose", "addiction",
    "mental breakdown", "can’t cope", "overwhelmed", "drinking problem", "need help",
    "lonely", "isolated", "hopeless", "I want to die", "crying all day", "self harm", "relapse"
]

# Build query for snscrape
query = "(" + " OR ".join(f'"{kw}"' for kw in keywords) + ") lang:en since:2023-01-01"

# Max number of tweets to extract
total_posts = 1000

posts_data = []

print(f"Scraping up to {total_posts} tweets...")

# Iterate over snscrape results
for idx, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    print(f"Extracting X Posts: {idx + 1}/{total_posts}", end='\r')  # Inline progress
    if idx >= total_posts:
        break

    posts_data.append({
        'id': tweet.id,
        'timestamp': tweet.date.isoformat(),
        'username': tweet.user.username,
        'content': tweet.content,
        'likes': tweet.likeCount,
        'retweets': tweet.retweetCount,
        'replies': tweet.replyCount
    })

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.lower()

df = pd.DataFrame(posts_data)
df['clean_content'] = df['content'].apply(clean_text)

# Sort rows by timestamp
df = df.sort_values(by='timestamp', ascending=False)

# Add timestamp to the file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"data/extracted_posts/extracted_posts_keywords_x_{timestamp}.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file}")
