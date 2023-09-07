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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('sentiment_analysis.csv')

# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_word(word):
    return lemmatizer.lemmatize(word)

text = ' '.join(data['review']).lower()
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

dominant_count = 5000
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

# Clustering and Train-Test Split
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data['review_title'])

# Assuming you have a 'labels' column in your data for true labels
y = data['review_title']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use KMeans clustering on the training data
kmeans = KMeans(n_clusters=9, random_state=7)  # assuming 5 clusters
kmeans.fit(X_train)

# Predict clusters for the training data
train_preds = kmeans.predict(X_train)

# Compute silhouette score for the training data
silhouette = silhouette_score(X_train, train_preds)
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Accuracy: {silhouette*100:.2f}%")
# Predict clusters for the test data
test_preds = kmeans.predict(X_test)

# Compute accuracy (NOTE: This might not be meaningful for clustering)
accuracy = accuracy_score(X_train, test_preds)
print(f"Accuracy_1:",accuracy)

# Compute accuracy (NOTE: This might not be meaningful for clustering)
accuracy = accuracy_score(y_test, test_preds)
print(f"Accuracy_2:",accuracy)
