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
    latest_file, _, timestamp = extracted_files[0]
    return latest_file, timestamp.strftime('%Y%m%d_%H%M%S')


def save_dataframe_with_metadata(df, output_file, metadata=None, copy_file=None):
    """
    Save a DataFrame to a CSV file with metadata at the top.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_file (str): The path to the output CSV file.
        metadata (dict): A dictionary of metadata to write as comments.
        copy_file (str, optional): If provided, saves a copy of the file to this path.
    """
    with open(output_file, 'w') as f:
        if metadata:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        df.to_csv(f, index=False)
    print(f"Saved: {output_file}")

    if copy_file:
        with open(copy_file, 'w') as f:
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
            df.to_csv(f, index=False)
        print(f"Copied to: {copy_file}")

def get_metadata_from_classified_file(classified_posts_file, key):
    """
    Extract the value of a specific metadata key from the classified posts file.

    Args:
        classified_posts_file (str): Path to the classified posts CSV file.
        key (str): The metadata key to extract (e.g., "Extracted posts file").

    Returns:
        str: The value of the metadata key, or None if the key is not found.
    """
    with open(classified_posts_file, 'r') as f:
        for line in f:
            if line.startswith(f"# {key}:"):
                # Extract the value after the colon
                return line.split(":", 1)[1].strip()
    return None