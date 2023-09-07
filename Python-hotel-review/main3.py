# Import necessary libraries and modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file into a pandas DataFrame
data = pd.read_csv('sentiment_analysis.csv')

# Clustering Step
# Initialize the TF-IDF Vectorizer with a maximum of 5000 features and to ignore English stop words
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Convert the 'customer_review' column to a matrix of TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(data['customer_review'])

# Choose the number of clusters for K-means clustering
num_clusters = 3

# Initialize the K-means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Apply clustering on the TF-IDF matrix and store the cluster labels in a new 'cluster' column
data['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Association Rule Mining Step
# One-hot encode the 'customer_rating' and 'hotel_name' columns for association rule mining
data_encoded = pd.get_dummies(data[['customer_rating', 'hotel_name']])

# Identify frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(data_encoded, min_support=0.1, use_colnames=True)

# Generate association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Classification Step
# Create a copy of the one-hot encoded data and add the 'cluster' column as a feature
X = data_encoded.copy()
X['cluster'] = data['cluster']

# Set the 'sentiment_type' column as the target variable
y = data['sentiment_type']

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict the sentiment types on the test data
predictions = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy_dt = accuracy_score(y_test, predictions)

# Print results
print("Clustering Results:")
print(data['cluster'].value_counts())

print("\nAssociation Rule Mining Results (Top 5 rules based on Lift):")
print(rules.sort_values(by='lift', ascending=False).head(5))

print("\nClassification Results using Decision Tree:")
print("Accuracy:", accuracy_dt)
print(classification_report(y_test, predictions))

# Note: To further optimize performance, one could consider testing other classifiers like RandomForest,
# GradientBoosting, SVM, etc., training them on the same training data, and evaluating on the test data.
# By comparing their accuracy values, one can determine the best performing model.
