import praw
import config
import os
from glob import glob
from datetime import datetime

def bot_login():
    print("Logging in...")
    reddit = praw.Reddit(username = config.reddit_username,
                    password = config.reddit_password,
                    client_id = config.reddit_client_id,
                    client_secret = config.reddit_client_secret,
                    user_agent = "crisis-monitoring-script")
    print("Logged in as {}".format(reddit.user.me()))
    return reddit

# Extract the timestamp from the filenames and find the latest one
def extract_timestamp(file):
    try:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        timestamp_str = filename.split('_')[-1].replace('.csv', '')
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except ValueError:
        return datetime.min  # Return a very old date if parsing fails

def extract_timestamp_and_number(file):
    try:
        filename = os.path.basename(file)
        parts = filename.replace('.csv', '').split('_')

        # Ensure the second part is numeric
        if not parts[1].isdigit():
            raise ValueError(f"Expected a numeric value, but got '{parts[1]}' in file: {file}")

        number = int(parts[1])  # '2949' from 'extracted_2949'
        timestamp_str = parts[-2] + '_' + parts[-1]  # '20250327_235605'
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

        return number, timestamp
    except (ValueError, IndexError) as e:
        print(f"Failed to parse file: {file}, Error: {e}")
        return 0, datetime.min


def get_latest_file(directory, prefix, specified_file=None):
    if specified_file:
        # If a specific file is provided, return it with its timestamp
        if not os.path.exists(specified_file):
            raise FileNotFoundError(f"Specified file '{specified_file}' does not exist.")
        return specified_file, datetime.fromtimestamp(os.path.getmtime(specified_file)).strftime('%Y%m%d_%H%M%S')

    # Glob all files matching the prefix pattern
    files = glob(os.path.join(directory, f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in directory '{directory}'")

    # Extract number and timestamp for all files
    extracted_files = []
    for file in files:
        number, timestamp = extract_timestamp_and_number(file)
        extracted_files.append((file, number, timestamp))

    # Sort by number (descending) and then by timestamp (descending)
    extracted_files.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Select the latest file
    latest_file, _, _ = extracted_files[0]
    formatted_time = datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y%m%d_%H%M%S')
    return latest_file, formatted_time