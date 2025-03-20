# Re-load necessary libraries after reset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

# Re-load the generated datasets
customer_profiles = pd.read_csv("./customer_profiles.csv")
social_media_activity = pd.read_csv("./social_media_activity.csv")
purchase_history = pd.read_csv("./purchase_history.csv")
sentiment_analysis = pd.read_csv("./sentiment_analysis.csv")

# Merge datasets on customer_id
customer_data = customer_profiles.merge(social_media_activity, on="customer_id")\
                                .merge(purchase_history, on="customer_id")\
                                .merge(sentiment_analysis, on="customer_id")

# Convert categorical variables to numerical encoding
customer_data["gender"] = customer_data["gender"].astype("category").cat.codes
customer_data["location"] = customer_data["location"].astype("category").cat.codes
customer_data["interests"] = customer_data["interests"].astype("category").cat.codes
customer_data["favorite_category"] = customer_data["favorite_category"].astype("category").cat.codes
customer_data["most_frequent_feedback"] = customer_data["most_frequent_feedback"].astype("category").cat.codes

# Select features for training the recommendation model
features = [
    "age", "gender", "location", "interests", "past_interactions",
    "likes", "shares", "posts", "engagement_score",
    "total_purchases", "avg_purchase_value", "favorite_category", "last_purchase_days_ago",
    "review_count", "avg_sentiment_score", "most_frequent_feedback"
]

# Normalize the data
scaler = StandardScaler()
customer_features = scaler.fit_transform(customer_data[features])

# Train a KNN model for recommendations using cosine similarity
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(customer_features)

def recommend_products_services(customer_id):
    """Recommend products/services based on similar customers using KNN with cosine similarity."""
    customer_index = customer_data[customer_data["customer_id"] == customer_id].index[0]
    distances, indices = knn.kneighbors([customer_features[customer_index]])
    
    # Get similar customers
    similar_customers = customer_data.iloc[indices[0][1:]]  # Exclude self
    
    # Recommend based on favorite categories of similar customers
    recommended_categories = similar_customers["favorite_category"].mode().tolist()
    
    # Recommend services based on engagement patterns
    recommended_services = []
    if similar_customers["engagement_score"].mean() > 5:
        recommended_services.append("Premium Membership")
    if similar_customers["likes"].mean() > 100:
        recommended_services.append("Exclusive Discounts")
    
    return {
        "Recommended Product Categories": recommended_categories,
        "Recommended Services": recommended_services
    }

# Naive Bayes for Sentiment Analysis
def train_sentiment_model():
    """Train a Naive Bayes model for sentiment classification."""
    X_train, X_test, y_train, y_test = train_test_split(
        sentiment_analysis["review_count"].astype(str),  # Convert to string for TF-IDF
        sentiment_analysis["most_frequent_feedback"],  # Target labels
        test_size=0.2, random_state=42
    )

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Evaluate the model
    y_pred = nb_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    return vectorizer, nb_model, accuracy

# Train the sentiment model
vectorizer, nb_model, sentiment_accuracy = train_sentiment_model()

def predict_sentiment(text):
    """Predict sentiment using the trained Naive Bayes model."""
    text_tfidf = vectorizer.transform([text])
    return nb_model.predict(text_tfidf)[0]

# Example usage
sample_customer = np.random.choice(customer_data["customer_id"])
recommendations = recommend_products_services(sample_customer)
predicted_sentiment = predict_sentiment("20")  # Example input for review count as string

# Output results
{
    "sample_customer": sample_customer,
    "recommendations": recommendations,
    "predicted_sentiment": predicted_sentiment,
    "sentiment_model_accuracy": sentiment_accuracy
}
