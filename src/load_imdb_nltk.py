import nltk
from nltk.corpus import movie_reviews
import random
import pandas as pd

# Load and shuffle file IDs
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# Convert to DataFrame
texts, labels = zip(*documents)
df = pd.DataFrame({'text': texts, 'label': labels})

# Check data
print(df.head())
print(df['label'].value_counts())

# Check total rows
print(f"Total reviews: {len(df)}")

# Show a random review and its label
sample = df.sample(1)
print("\n--- Sample Review ---")
print("Label:", sample['label'].values[0])
print("Text:", sample['text'].values[0][:500], "...")  # Only show the first 500 chars

