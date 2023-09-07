import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words
import pycountry
import nltk

# Load the dataset
data = pd.read_csv('sentiment_analysis.csv')

# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to lemmatize words
def lemmatize_word(word):
    return lemmatizer.lemmatize(word)

# Combine all reviews into one large text
text = ' '.join(data['customer_review']).lower()

# Split it into words
words_list = re.findall(r'\w+', text)

# Lemmatize the words
lemmatized_words = [lemmatize_word(word) for word in words_list]

# Get word frequencies
word_freq = Counter(lemmatized_words)

# Define and update stopwords
stopwords = set(STOPWORDS)
custom_stopwords = {"some", "other", "stopwords", "dubai"}  # Add custom stopwords if any
stopwords.update(custom_stopwords)

# Get known dictionary words
known_words = set(words.words())

# Extract all country names
country_names = {country.name.lower() for country in pycountry.countries}

# POS tagging to detect proper nouns (NNP) and nouns (NN)
tagged = nltk.pos_tag(word_freq.keys())
propernouns = [word for word, pos in tagged if pos == 'NNP' or pos == 'NN']

# Remove stopwords, known dictionary words, country names, proper nouns, and nouns from the Counter object
for word in list(word_freq):
    if word in stopwords or word in known_words or word in country_names or word in propernouns:
        del word_freq[word]

# Number of dominant words you want to display
dominant_count = 1000

# Get the most common words
dominant_words = dict(word_freq.most_common(dominant_count))

# Generate word cloud
wordcloud = WordCloud(width=1000, height=500, background_color='white', stopwords=stopwords).generate_from_frequencies(dominant_words)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Print word counts
for word, count in dominant_words.items():
    print(f"{word}: {count}")
