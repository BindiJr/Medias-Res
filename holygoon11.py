import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Load the CSV file containing your movies data
df = pd.read_csv("C:/TMDBGoon/tmdb_top_rated_movies.csv")

# Preprocess Textual Data (overview, tagline, keywords) with n-grams and weighting
def preprocess_text_data(df):
    # Use 1-gram, 2-gram, and 3-gram to extract more contextual features
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=2000)
    overview_matrix = vectorizer.fit_transform(df['overview'].fillna(''))
    tagline_matrix = vectorizer.transform(df['tagline'].fillna(''))
    keywords_matrix = vectorizer.transform(df['keywords'].fillna(''))
    
    return overview_matrix, tagline_matrix, keywords_matrix

# Preprocess Genres (Multi-label binarization)
def preprocess_genres(df):
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(df['genres'])
    return genres_matrix

# Preprocess Actors (Multi-label binarization)
def preprocess_actors(df):
    mlb = MultiLabelBinarizer()
    actors_matrix = mlb.fit_transform(df['actors'])
    return actors_matrix

# Preprocess Producers (Multi-label binarization)
def preprocess_producers(df):
    mlb = MultiLabelBinarizer()
    producers_matrix = mlb.fit_transform(df['producers'])
    return producers_matrix

# Scale numerical features
def preprocess_numerical_data(df):
    numerical_columns = ['vote_average', 'popularity', 'budget', 'revenue']
    numerical_data = df[numerical_columns].fillna(0).values
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)
    return numerical_data_scaled

# Function to combine all features (textual, categorical, numerical)
def combine_features(df):
    overview_matrix, tagline_matrix, keywords_matrix = preprocess_text_data(df)
    genres_matrix = preprocess_genres(df)
    actors_matrix = preprocess_actors(df)
    producers_matrix = preprocess_producers(df)
    numerical_data_scaled = preprocess_numerical_data(df)

    # Combine all feature matrices into one
    combined_features = np.hstack([
        overview_matrix.toarray(),
        tagline_matrix.toarray(),
        keywords_matrix.toarray(),
        genres_matrix,
        actors_matrix,
        producers_matrix,
        numerical_data_scaled
    ])
    
    return combined_features

# Dimensionality Reduction with TruncatedSVD (works better for sparse data like TF-IDF)
def apply_dimensionality_reduction(features):
    svd = TruncatedSVD(n_components=100)
    reduced_features = svd.fit_transform(features)
    return reduced_features

# Function to compute the cosine similarity matrix
def compute_similarity_matrix(features):
    similarity_matrix = cosine_similarity(features)
    return similarity_matrix

# Function to get the top N most similar movies to a given movie
def get_similar_movies(movie_index, similarity_matrix, top_n=5):
    similarity_scores = similarity_matrix[movie_index]
    similar_movie_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    return similar_movie_indices

# Function to recommend similar movies based on user input
def recommend_similar_movies(df):
    # Combine and reduce features
    combined_features = combine_features(df)
    reduced_features = apply_dimensionality_reduction(combined_features)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(reduced_features)
    
    # Display a list of movie titles
    print("Available movies:")
    for i, title in enumerate(df['title'].head(20)):  # Display first 20 movie titles
        print(f"{i}: {title}")
    
    # Ask user to input the index of the movie
    try:
        movie_index = int(input("Enter the index number of the movie you want to find similar films to: "))
        
        # Get top 5 similar movies
        similar_movie_indices = get_similar_movies(movie_index, similarity_matrix, top_n=5)
        
        # Print the recommended similar movies
        print(f"\nMovies similar to '{df.iloc[movie_index]['title']}':")
        for idx in similar_movie_indices:
            print(df.iloc[idx]['title'])
    except (ValueError, IndexError):
        print("Invalid input. Please make sure you select a valid movie index.")

# Run the recommend_similar_movies function to allow user interaction
recommend_similar_movies(df)
