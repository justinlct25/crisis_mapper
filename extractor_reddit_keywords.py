# extract_posts.py
import pandas as pd
from datetime import datetime
import re
import helper

# Setup Reddit API client
reddit = helper.bot_login()

# Target subreddits where users often discuss distress/crisis
subreddits = [
    "depression", "SuicideWatch", "mentalhealth", "offmychest",
    "anxiety", "Needafriend", "depression_help", "KindVoice"
]

# Keywords to filter for relevance
keywords = [
    "depressed", "anxiety", "panic attack", "suicidal", "overdose", "addiction",
    "mental breakdown", "can’t cope", "overwhelmed", "drinking problem", "need help",
    "lonely", "isolated", "hopeless", "I want to die", "crying all day", "self harm", "relapse"
]

posts_data = []
total_target = 5000  # Approximate number of posts to collect

print(f"Extracting up to {total_target} posts from mental health-related subreddits...")

for sub_name in subreddits:
    subreddit = reddit.subreddit(sub_name)
    count = 0
    for submission in subreddit.new(limit=None):  # Use .new() to get recent posts
        if submission.stickied or submission.removed_by_category:
            continue  # skip pinned or removed posts
        combined_text = (submission.title or "") + " " + (submission.selftext or "")
        if any(kw.lower() in combined_text.lower() for kw in keywords):
            posts_data.append({
                'subreddit': sub_name,
                'id': submission.id,
                'timestamp': datetime.utcfromtimestamp(submission.created_utc),
                'content': combined_text,
                'author': str(submission.author),
                'upvotes': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url
            })
            count += 1
        if len(posts_data) >= total_target:
            break
        print(f"Fetching from r/{sub_name}: {count} posts", end='\r')
    print(f"Fetched from r/{sub_name}: {count} posts")
    if len(posts_data) >= total_target:
        break

# Clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

df = pd.DataFrame(posts_data)
df['clean_content'] = df['content'].apply(clean_text)

# Sort rows by timestamp
df = df.sort_values(by='timestamp', ascending=False)

# Add timestamp to the file name
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f"data/extracted_posts/extracted_posts_reddit_keywords_{timestamp}.csv"
df.to_csv(output_file, index=False)
print(f"Saved: {output_file}")
