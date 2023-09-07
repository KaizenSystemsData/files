import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Sample dataset
data = {
    'review': [
        'The product is amazing and I love it',
        'Really bad experience, will not buy again',
        'It is okay, not great but not bad',
        'Absolutely love this, great purchase',
        'Terrible product, wasted money'
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Data Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model Selection and Training: Support Vector Machine
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluation
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)

# Predicting sentiment of new reviews using the trained SVM model
new_reviews = [
    "I'm so happy with this purchase!",
    "Worst product ever.",
    "It's decent for the price."
]

X_new = vectorizer.transform(new_reviews)
predicted_sentiments_svm = svm.predict(X_new)

print("\nPredicted Sentiments for New Reviews:")
for review, sentiment in zip(new_reviews, predicted_sentiments_svm):
    print(f"Review: '{review}' - Predicted Sentiment: '{sentiment}'")
