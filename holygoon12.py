import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("C:/TMDBGoon/tmdb_top_rated_movies.csv")

# Preprocess genres to combine genre columns
def preprocess_genres(df):
    # Combine all genre columns into a single list for each movie
    genres = df[['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']].apply(lambda row: [str(val) for val in row if pd.notnull(val)], axis=1)
    
    # Use MultiLabelBinarizer to one-hot encode the genres
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(genres)
    
    return genres_matrix

# Preprocess the textual data (overview, tagline, and keywords)
def preprocess_text_data(df):
    # Create TF-IDF vectorizer for overview, tagline, and keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    overview_matrix = vectorizer.fit_transform(df['overview'].fillna(''))
    tagline_matrix = vectorizer.fit_transform(df['tagline'].fillna(''))
    keywords_matrix = vectorizer.fit_transform(df['keywords'].fillna(''))

    return overview_matrix, tagline_matrix, keywords_matrix

# Combine features (genres, overview, tagline, keywords)
def combine_features(df):
    genres_matrix = preprocess_genres(df)
    overview_matrix, tagline_matrix, keywords_matrix = preprocess_text_data(df)
    
    # Combine all the feature matrices
    combined_matrix = hstack([genres_matrix, overview_matrix, tagline_matrix, keywords_matrix])
    
    return combined_matrix

# Function to recommend similar movies
def recommend_similar_movies(df):
    combined_features = combine_features(df)
    
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(combined_features)
    
    # Get movie recommendations for the first movie
    movie_idx = 0  # Example: Recommend movies similar to the first movie
    similar_movies = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort the movies by similarity score in descending order
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    
    # Display the top 5 similar movies (excluding the movie itself)
    print("Top 5 similar movies to '{}'".format(df.iloc[movie_idx]['title']))
    for idx, score in similar_movies[1:6]:
        print(f"{df.iloc[idx]['title']} (Similarity Score: {score:.3f})")

# Run the recommendation function
recommend_similar_movies(df)
