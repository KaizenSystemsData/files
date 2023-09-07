import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words
import pycountry
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the dataset
data = pd.read_csv('sentiment_analysis.csv')

# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def lemmatize_word(word):
    return lemmatizer.lemmatize(word)


text = ' '.join(data['customer_review']).lower()
words_list = re.findall(r'\w+', text)
lemmatized_words = [lemmatize_word(word) for word in words_list]
word_freq = Counter(lemmatized_words)

stopwords = set(STOPWORDS)
custom_stopwords = {"some", "other", "stopwords", "dubai"}
stopwords.update(custom_stopwords)

known_words = set(words.words())
country_names = {country.name.lower() for country in pycountry.countries}

tagged = nltk.pos_tag(word_freq.keys())
propernouns = [word for word, pos in tagged if pos == 'NNP' or pos == 'NN']

for word in list(word_freq):
    if word in stopwords or word in known_words or word in country_names or word in propernouns:
        del word_freq[word]

dominant_count = 1000
dominant_words = dict(word_freq.most_common(dominant_count))

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()
sentiments = {'positive': [], 'neutral': [], 'negative': []}

for word, count in dominant_words.items():
    score = sia.polarity_scores(word)['compound']
    if score > 0.05:
        sentiments['positive'].append(word)
    elif score < -0.05:
        sentiments['negative'].append(word)
    else:
        sentiments['neutral'].append(word)

for sentiment, words_list in sentiments.items():
    freqs = {word: dominant_words[word] for word in words_list}
    wordcloud = WordCloud(width=1000, height=500, background_color='white',
                          stopwords=stopwords).generate_from_frequencies(freqs)

    plt.figure(figsize=(10, 5))
    plt.title(sentiment)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    for word in words_list:
        print(f"{word}: {dominant_words[word]}")