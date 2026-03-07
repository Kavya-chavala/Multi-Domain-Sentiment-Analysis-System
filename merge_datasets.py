import pandas as pd

# ---------- Load IMDB ----------
imdb = pd.read_csv("sentiment_dataset.csv")

# Keep only needed columns
imdb = imdb[['review', 'sentiment']]

# ---------- Load Twitter ----------
twitter = pd.read_csv("twitter_training.csv", header=None)

# Assign proper column names
twitter.columns = ['id', 'entity', 'sentiment', 'review']

# Keep only review & sentiment
twitter = twitter[['review', 'sentiment']]

# Convert sentiment to lowercase
twitter['sentiment'] = twitter['sentiment'].str.lower()

# Keep only positive and negative
twitter = twitter[twitter['sentiment'].isin(['positive', 'negative'])]

# ---------- Balance Sampling ----------
# Take equal samples from both datasets
imdb_sample = imdb.sample(n=30000, random_state=42)
twitter_sample = twitter.sample(n=30000, random_state=42)

# Combine
combined = pd.concat([imdb_sample, twitter_sample], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1, random_state=42)

# Save merged dataset
combined.to_csv("merged_dataset.csv", index=False)

print("✅ Datasets merged successfully!")
print("Total rows:", len(combined))
print("\nClass Distribution:")
print(combined['sentiment'].value_counts())
