import praw
import config
import os
from glob import glob

def bot_login():
    print("Logging in...")
    reddit = praw.Reddit(username = config.reddit_username,
                    password = config.reddit_password,
                    client_id = config.reddit_client_id,
                    client_secret = config.reddit_client_secret,
                    user_agent = "crisis-monitoring-script")
    print("Logged in as {}".format(reddit.user.me()))
    return reddit

# Find the latest file from post_extractor_keywords.py
def get_latest_file(directory, prefix):
    files = glob(os.path.join(directory, f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in directory '{directory}'")
    latest_file = max(files, key=os.path.getctime)
    latest_time = os.path.getctime(latest_file)  # Get creation time
    return latest_file, latest_time
