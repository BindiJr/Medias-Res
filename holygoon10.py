import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the movie data (update the path)
df = pd.read_csv("C:\\TMDBGoon\\tmdb_top_rated_movies.csv")

# Assuming the rest of your code follows below...

# Function to clean and prepare data
def clean_and_extract_columns(df):
    genre_columns = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']
    df['genres'] = df[genre_columns].apply(lambda x: set(x.dropna()), axis=1)

    actor_columns = ['actor_1', 'actor_2', 'actor_3', 'actor_4', 'actor_5', 'actor_6']
    df['actors'] = df[actor_columns].apply(lambda x: set(x.dropna()), axis=1)

    producer_columns = ['producer_1', 'producer_2', 'producer_3']
    df['producers'] = df[producer_columns].apply(lambda x: set(x.dropna()), axis=1)

    df['same_collection'] = df['collection_name'].apply(lambda x: 1 if pd.notna(x) else 0)
    df['budget'] = df['budget'].fillna(0)
    df['revenue'] = df['revenue'].fillna(0)

    return df

# Preprocess Textual Data (overview, tagline, keywords)
def preprocess_text_data(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    overview_matrix = vectorizer.fit_transform(df['overview'].fillna(''))
    tagline_matrix = vectorizer.transform(df['tagline'].fillna(''))
    keywords_matrix = vectorizer.transform(df['keywords'].fillna(''))
    
    return overview_matrix, tagline_matrix, keywords_matrix

# Preprocess Categorical Data (genres, actors, producers)
def preprocess_categorical_data(df):
    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(df['genres'])
    actors_matrix = mlb.fit_transform(df['actors'])
    producers_matrix = mlb.fit_transform(df['producers'])

    return genres_matrix, actors_matrix, producers_matrix

# Combine all features (textual + categorical + numerical)
def combine_all_features(df, overview_matrix, tagline_matrix, keywords_matrix, genres_matrix, actors_matrix, producers_matrix):
    numerical_columns = ['vote_average', 'popularity', 'budget', 'revenue']
    numerical_data = df[numerical_columns].fillna(0).values

    combined_features = np.hstack([
        overview_matrix.toarray(),
        tagline_matrix.toarray(),
        keywords_matrix.toarray(),
        genres_matrix,
        actors_matrix,
        producers_matrix,
        df[['same_collection']].values,
        numerical_data
    ])

    return combined_features

# Reduce dimensionality with PCA
def reduce_dimensionality(combined_features):
    pca = PCA(n_components=100)  # Reduce to 100 dimensions
    reduced_features = pca.fit_transform(combined_features)
    return reduced_features

# Compute Cosine Similarity
def compute_similarity(reduced_features):
    similarity_matrix = cosine_similarity(reduced_features)
    return similarity_matrix

# Get the top N most similar movies
def get_similar_movies(movie_index, similarity_matrix, df, top_n=5):
    similarity_scores = similarity_matrix[movie_index]
    similar_movie_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    similar_movie_titles = df.iloc[similar_movie_indices]['title'].values
    return similar_movie_titles

# Main recommendation function
def recommend_similar_movies(df):
    df = clean_and_extract_columns(df)

    # Preprocess textual data
    overview_matrix, tagline_matrix, keywords_matrix = preprocess_text_data(df)

    # Preprocess categorical data
    genres_matrix, actors_matrix, producers_matrix = preprocess_categorical_data(df)

    # Combine all features
    combined_features = combine_all_features(df, overview_matrix, tagline_matrix, keywords_matrix, genres_matrix, actors_matrix, producers_matrix)

    # Apply PCA to reduce dimensionality
    reduced_features = reduce_dimensionality(combined_features)

    # Compute similarity matrix
    similarity_matrix = compute_similarity(reduced_features)

    # Let the user select a movie and get recommendations
    print("Available movies:")
    for i, title in enumerate(df['title'].head(20)):
        print(f"{i}: {title}")

    try:
        movie_index = int(input("Enter the index number of the movie you want to find similar films to: "))
        recommended_titles = get_similar_movies(movie_index, similarity_matrix, df, top_n=5)

        print(f"\nMovies similar to '{df.iloc[movie_index]['title']}':")
        for title in recommended_titles:
            print(title)

    except (ValueError, IndexError):
        print("Invalid input. Please make sure you select a valid movie index.")

# Assuming you already have the movie data loaded in 'df'
recommend_similar_movies(df)
