# extract_posts.py
import pandas as pd
from datetime import datetime
import re
import helper
from prawcore.exceptions import RequestException
import time

def extract_reddit_posts_keywords(extracted_posts_csv=None):
    reddit = helper.bot_login()

    subreddits = [
        "depression", "SuicideWatch", "mentalhealth", "offmychest",
        "anxiety", "Needafriend", "depression_help", "KindVoice",
        "TrueOffMyChest", "confessions", "Vent", "Rant",
        "ptsd", "CPTSD", "BPD", "MentalHealthSupport", "DecidingToBeBetter", "Anxietyhelp",
        "ForeverAlone", "SocialAnxiety", "lonely",
        "sad", "teenagers", "darkjokes", "DeadBedrooms",
        "mentalillness", "AnxietyDepression", "traumatoolbox",
        "LifeAfterNarcissism", "BreakUps", "Divorce", "GriefSupport", "ChronicPain",
        "stopdrinking", "OpiatesRecovery", "Leaves", "REDDITORSINRECOVERY",
        "TeenagersAdvice", "HighSchool", "College",
        "Lonely", "ForeverAloneDating", "Introvert",
        "UnsentLetters", "JustVent"
    ]

    keywords = [
        "depressed", "anxiety", "panic attack", "suicidal", "overdose", "addiction",
        "mental breakdown", "can’t cope", "overwhelmed", "drinking problem", "need help",
        "lonely", "isolated", "hopeless", "I want to die", "crying all day", "self harm", "relapse", "kill myself",
        "want to kill myself", "going to kill myself", "planning to end it all", "ending it tonight",
        "taking my life", "thinking about ending it", "final goodbye", "this is my last post",
        "tired of living", "life is meaningless", "I give up", "no reason to go on", "I'm done",
        "everything hurts", "it’s over", "ready to go", "razor", "pills", "hanging", "jumping",
        "carbon monoxide", "bridge", "I can’t do this anymore", "I don’t want to be here", "I just want it to stop",
        "no point in living", "ending everything", "ready to die", "giving up", "goodbye everyone",
        "saying goodbye", "this is it", "hope it's quick", "final post", "numb", "broken inside",
        "everything is too much", "can’t take it anymore", "I’m worthless", "nothing matters",
        "exhausted with life", "just want peace", "want to disappear", "nobody cares",
        "kms", "kys", "unalive", "rope", "off myself"
    ]

    posts_data = []
    total_target = 5000 

    # Load the last extracted posts CSV file if provided
    existing_ids = set()
    original_df = pd.DataFrame()  
    try:
        latest_file, latest_time_formatted = helper.get_latest_file('data/extracted_posts', 'extracted', extracted_posts_csv)
        print(f"Loading latest classified file: {latest_file}")
        original_df = pd.read_csv(latest_file)
        existing_ids = set(original_df['id'])
        print(f"Loaded {len(existing_ids)} existing post IDs from {latest_file}")
    except FileNotFoundError:
        print(f"File {latest_file} not found. Proceeding without filtering existing posts.")

    print(f"Extracting up to {total_target} posts from mental health-related subreddits...")

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            next(subreddit.new(limit=1)) # trigger a fetch to check whether the subreddit exists
        except Exception as e:
            print(f"Skipping subreddit '{sub_name}' due to error: {e}")
            continue

        subreddit = reddit.subreddit(sub_name)
        count = 0

        # Retry fetching submissions up to 3 times
        for attempt in range(3): 
            try:
                for submission in subreddit.new(limit=None):  # Get recent posts
                    if submission.stickied or submission.removed_by_category:   # Skip pinned or removed posts
                        continue  
                    if submission.id in existing_ids:   # Skip posts that already exist in the last extracted file
                        continue  
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
                break 
            except RequestException as e:
                print(f"Request failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)

        print(f"Fetched from r/{sub_name}: {count} posts")
        if len(posts_data) >= total_target:
            break

    def clean_text(text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    new_posts_df = pd.DataFrame(posts_data)
    if not new_posts_df.empty:
        new_posts_df['clean_content'] = new_posts_df['content'].apply(clean_text)

    # Combine the original already extracted posts and newly extracted posts, removing duplicates
    combined_df = pd.concat([original_df, new_posts_df]).drop_duplicates(subset='id', keep='last')

    # Sort rows by timestamp
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    combined_df = combined_df.sort_values(by='timestamp', ascending=False)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data/extracted_posts/extracted_{len(combined_df)}_reddit_posts_by_keywords_{timestamp}.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    extract_reddit_posts_keywords()