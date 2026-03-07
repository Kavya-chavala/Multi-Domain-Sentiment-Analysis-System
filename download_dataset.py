import pandas as pd

url = "https://raw.githubusercontent.com/dD2405/Twitter-Sentiment-Analysis/master/train.csv"

print("Downloading dataset...")

data = pd.read_csv(url)

# Keep only necessary columns
data = data[['text', 'airline_sentiment']]

# Rename columns
data.columns = ['review', 'sentiment']

# Keep only 3 classes
data = data[data['sentiment'].isin(['positive', 'neutral', 'negative'])]

# Sample smaller subset for speed (6000 rows)
data = data.sample(n=6000, random_state=42)

data.to_csv("sentiment_dataset.csv", index=False)

print("✅ 3-Class Twitter dataset saved successfully!")
print("Total rows:", len(data))
