# Consolidated Sentiment Analysis with Word Clouds

import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Review': [
        'The room was clean and spacious.',
        'I loved the breakfast they offered!',
        'The staff was rude. Not coming back.',
        'Location was great, right in the city center.',
        'The bed was uncomfortable.',
        'Service was excellent!',
        'Bathroom was dirty when we arrived.',
        'View from the room was amazing.',
        'Wifi connection was terrible.',
        'Overall, a pleasant stay.'
    ],
    'Sentiment': [
        'positive',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive'
    ]
}
#df = pd.DataFrame(data)
df =pd.read_csv('sentiment_analysis.csv')
# Preprocess
def manual_preprocess(text):
    text = text.lower()
    tokens = text.split()
    manual_stopwords = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'in', 'for', 'i', 'it', 'on', 'is', 'this', 'that', 'my', 'at', 'not', 'but', 'from', 'we', 'they']
    tokens = [word for word in tokens if word not in manual_stopwords and word not in string.punctuation]
    return " ".join(tokens)

df['Processed_Review'] = df['Review'].apply(manual_preprocess)

# Feature Extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Processed_Review'])

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.3, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Word Clouds
positive_text = " ".join(df[df['Sentiment'] == 'positive']['Processed_Review'].values)
negative_text = " ".join(df[df['Sentiment'] == 'negative']['Processed_Review'].values)
positive_wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='viridis').generate(positive_text)
negative_wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='inferno').generate(negative_text)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(positive_wordcloud, interpolation='bilinear')
ax[0].axis('off')
ax[0].set_title('Positive Reviews')
ax[1].imshow(negative_wordcloud, interpolation='bilinear')
ax[1].axis('off')
ax[1].set_title('Negative Reviews')
plt.tight_layout()
plt.show()

print(f"Model Accuracy: {accuracy * 100:.2f}%")
