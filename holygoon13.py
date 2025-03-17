import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

# Function to preprocess genres
def preprocess_genres(df):
    # Combine all genre columns into a single list for each movie
    genres = df[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']].apply(lambda row: [str(val) for val in row if pd.notnull(val)], axis=1)
    
    # Use MultiLabelBinarizer to one-hot encode the genres
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(genres)
    
    return genres_matrix

# Function to preprocess text data
def preprocess_text_data(df):
    # Using TF-IDF Vectorizer for the overview, tagline, and keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Combine the three text features into a single column for processing
    combined_text = df['overview'].fillna('') + ' ' + df['tagline'].fillna('') + ' ' + df['keywords'].fillna('')
    
    text_matrix = vectorizer.fit_transform(combined_text)
    
    return text_matrix

# Combine the features into a single matrix
def combine_features(df):
    # Preprocess genres
    genres_matrix = preprocess_genres(df)
    
    # Preprocess text data
    text_matrix = preprocess_text_data(df)
    
    # Concatenate the genre and text matrices into one feature matrix
    combined_features = np.hstack((genres_matrix, text_matrix.toarray()))
    
    return combined_features

# Function to recommend similar movies based on movie_id
def recommend_similar_movies(df, movie_id):
    # Combine features
    combined_features = combine_features(df)
    
    # Calculate the similarity matrix
    similarity_matrix = cosine_similarity(combined_features)
    
    # Get the index of the movie from the movie_id
    movie_idx = df[df['id'] == movie_id].index[0]
    
    # Get pairwise similarity scores for the movie
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort the movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 most similar movies
    top_10_similar_movies = similarity_scores[1:11]
    
    print(f"Top 10 movies similar to movie ID {movie_id}:\n")
    for idx, score in top_10_similar_movies:
        print(f"Movie: {df.iloc[idx]['title']} - Similarity Score: {score}")

# Load the movie data CSV
df = pd.read_csv("C:/TMDBGoon/tmdb_top_rated_movies.csv")

# Test the recommendation with a sample movie ID (e.g., 1, or any valid movie ID from your dataset)
recommend_similar_movies(df, 1)  # Replace with a valid movie ID from your dataset
