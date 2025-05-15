import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt_tab')

# Load the dataset
from load_imdb_nltk import df  # import df from your previous script

# Download extra NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Clean a single text
def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join(char for char in text if char not in string.punctuation)  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]  # remove stopwords, non-alphabetic
    return ' '.join(filtered)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Show a sample cleaned text
print("\nOriginal:")
print(df['text'].iloc[0][:300])
print("\nCleaned:")
print(df['clean_text'].iloc[0][:300])

# Convert to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

# Labels (y)
y = df['label'].map({'pos': 1, 'neg': 0})

# Check shapes
print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

