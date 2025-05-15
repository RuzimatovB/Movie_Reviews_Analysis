import joblib
from preprocess_and_vectorize import vectorizer
from train_and_evaluate import model

# Save model and vectorizer
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved!")
