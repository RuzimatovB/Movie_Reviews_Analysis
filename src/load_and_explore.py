from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import matplotlib.pyplot as plt

# Load a subset of the dataset (for binary classification)
categories = ['rec.sport.hockey', 'sci.space']
data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Convert to DataFrame for easier exploration
df = pd.DataFrame({'text': data.data, 'label': data.target})
df['label'] = df['label'].map({0: 'hockey', 1: 'space'})  # Label names

# Show some rows
print(df.head())

# Plot class balance
df['label'].value_counts().plot(kind='bar', title='Class Distribution')
plt.show()

import nltk
nltk.download('movie_reviews')
nltk.download('punkt_tab')