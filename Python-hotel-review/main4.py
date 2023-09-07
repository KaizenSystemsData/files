# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv('sentiment_analysis.csv')

# Generate word cloud for customer reviews
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=set(['english']),
                      min_font_size=10).generate(' '.join(data['customer_review']))

# Plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Extract and display words from the customer reviews and their counts
word_list = ' '.join(data['customer_review']).split()
word_counts = pd.Series(word_list).value_counts()
print("Words and their counts:\n", word_counts)


# Display the total word count
total_words = sum(data['customer_review'].apply(lambda x: len(x.split())))
print(f"Total word count in the dataset: {total_words}")

# Convert customer reviews to TF-IDF matrix for numerical representation
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['customer_review'])

# Use KMeans for clustering reviews based on their TF-IDF representation
kmeans = KMeans(n_clusters=3, random_state=41)
data['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Convert categorical data to one-hot encoded format for association rule mining
data_encoded = pd.get_dummies(data[['customer_rating', 'hotel_name']])
frequent_itemsets = apriori(data_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Prepare data for classification
X = data_encoded.copy()
X['cluster'] = data['cluster']
y = data['sentiment_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)

# Random Forest Classifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)

# Define base models for stacking classifier
estimators = [
    ('dt', clf_dt),
    ('rf', clf_rf),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]

# Train stacking classifier
clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
clf_stack.fit(X_train, y_train)
pred_stack = clf_stack.predict(X_test)

# Display confusion matrix for the stacking classifier
conf_matrix = confusion_matrix(y_test, pred_stack)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Stacking Classifier')
plt.show()
print("\nClassification Report:\n", classification_report(y_test, pred_stack))

# ROC Curve for multi-class classification
y_test_bin = label_binarize(y_test, classes=list(y.unique()))
pred_stack_bin = clf_stack.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(y.unique())):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred_stack_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
for i, label in enumerate(y.unique()):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {label} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stacking Classifier')
plt.legend(loc='best')
plt.show()
