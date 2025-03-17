import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

# Function to preprocess genres
def preprocess_genres(df):
    genres = df[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']].apply(lambda row: [str(val) for val in row if pd.notnull(val)], axis=1)
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(genres)
    return genres_matrix

# Function to preprocess text data
def preprocess_text_data(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    combined_text = df['overview'].fillna('') + ' ' + df['tagline'].fillna('') + ' ' + df['keywords'].fillna('')
    text_matrix = vectorizer.fit_transform(combined_text)
    return text_matrix

# Combine the features into a single matrix
def combine_features(df):
    genres_matrix = preprocess_genres(df)
    text_matrix = preprocess_text_data(df)
    combined_features = np.hstack((genres_matrix, text_matrix.toarray()))
    return combined_features

# Function to recommend similar movies based on movie_id
def recommend_similar_movies(df, movie_id):
    # Check if the movie_id exists in the dataset
    if movie_id not in df['id'].values:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return
    
    combined_features = combine_features(df)
    similarity_matrix = cosine_similarity(combined_features)
    
    # Get the index of the movie from the movie_id
    movie_idx = df[df['id'] == movie_id].index[0]
    
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_10_similar_movies = similarity_scores[1:11]
    
    print(f"Top 10 movies similar to movie ID {movie_id}:\n")
    for idx, score in top_10_similar_movies:
        print(f"Movie: {df.iloc[idx]['title']} - Similarity Score: {score}")

# Load the movie data CSV
df = pd.read_csv("C:/TMDBGoon/tmdb_top_rated_movies.csv")

# Test with a valid movie ID (e.g., replace 1 with a valid ID)
recommend_similar_movies(df, 1)
