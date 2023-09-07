# Hotel Reviews Clustering Script

# Import necessary libraries
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Define the sample dataset
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
    ]
}
df = pd.DataFrame(data)

# Preprocess the reviews
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    manual_stopwords = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'in', 'for', 'i', 'it', 'on', 'is', 'this', 'that', 'my', 'at', 'not', 'but', 'from', 'we', 'they']
    tokens = [word for word in tokens if word not in manual_stopwords and word not in string.punctuation]
    return " ".join(tokens)

df['Processed_Review'] = df['Review'].apply(preprocess)

# Extract features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['Processed_Review'])

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_tfidf)

# Visualize clusters using t-SNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df['Cluster'], palette='viridis', s=100)
plt.title('t-SNE visualization of review clusters')
plt.show()

# Calculate and print the silhouette score
silhouette_avg = silhouette_score(X_tfidf, df['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Display the reviews with their assigned clusters
print(df[['Review', 'Cluster']])
