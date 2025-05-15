import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))

# Text cleaning function (same as used in preprocessing)
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered)

# Try predicting on a custom review
def predict_sentiment(review):
    clean = clean_text(review)
    features = vectorizer.transform([clean])
    prediction = model.predict(features)[0]
    return "positive" if prediction == 1 else "negative"

# Example
if __name__ == "__main__":
    sample_review = input("Enter a movie review: ")
    sentiment = predict_sentiment(sample_review)
    print("Sentiment:", sentiment)
