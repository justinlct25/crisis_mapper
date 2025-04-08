# classify_risk.py
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# import nltk
# nltk.download('vader_lexicon')

df = pd.read_csv('data/filtered_posts.csv')
sia = SentimentIntensityAnalyzer()

def classify_risk(text):
    if any(term in text for term in ["i want to die", "don’t want to be here", "kill myself"]):
        return "High"
    elif any(term in text for term in ["i feel lost", "need help", "no purpose"]):
        return "Moderate"
    else:
        return "Low"

# Add progress tracking
total_rows = len(df)
for idx, row in df.iterrows():
    df.at[idx, 'sentiment'] = sia.polarity_scores(row['clean_content'])['compound']
    df.at[idx, 'risk_level'] = classify_risk(row['clean_content'])
    print(f"Processing: {idx + 1}/{total_rows}", end='\r')  # Inline progress

# Reorder columns to make 'sentiment' and 'risk_level' the second and third columns
columns = ['sentiment', 'risk_level'] + [col for col in df.columns if col not in ['sentiment', 'risk_level']]
df = df[columns]

df.to_csv('data/classified_posts_keywords.csv', index=False)
print("Saved: data/classified_posts_keywords.csv")