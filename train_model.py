import pandas as pd
import joblib
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load IMDB&Twitter dataset
data = pd.read_csv("merged_dataset.csv")


# Load stopwords once



# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

print("Cleaning text...")
data["cleaned_review"] = data["review"].apply(clean_text)

X = data["cleaned_review"]
y = data["sentiment"]

print("Applying TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    lowercase=True
)


X_vectorized = vectorizer.fit_transform(X)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

print("Evaluating model...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ IMDB & Twitter Sentiment Model trained and saved successfully!")
