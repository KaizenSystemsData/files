import pandas as pd
from sklearn.pipeline import make_pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words
import pycountry
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

# Load the dataset
data = pd.read_csv('sentiment_analysis.csv')

# Use a fraction of the dataset
data = data.sample(frac=0.5, random_state=42)  # Adjust frac as needed

# Create an instance of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_word(word):
    return lemmatizer.lemmatize(word)

# ... [rest of your preprocessing and visualization code remains unchanged]

# Splitting the data for training
X = data['customer_review']
y = data['customer_review']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a sentiment analysis model with 'saga' solver
model = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='saga', max_iter=10000))
model.fit(X_train, y_train)

# Calculate and print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Visualize the distribution of actual vs. predicted sentiments
df_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(data=df_results, x='Actual')
plt.title('Actual Sentiments')

plt.subplot(1, 2, 2)
sns.countplot(data=df_results, x='Predicted')
plt.title('Predicted Sentiments')

plt.tight_layout()
plt.show()
