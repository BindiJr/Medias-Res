import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np

# Load the CSV file containing your movies data
df = pd.read_csv('tmdb_top_rated_movies.csv')

# Check if the required columns are present in the dataframe
required_columns = ['title', 'overview', 'genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6', 'vote_average', 'popularity', 'budget', 'revenue', 'actor_1', 'actor_2', 'actor_3', 'actor_4', 'actor_5', 'actor_6', 'producer_1', 'producer_2', 'producer_3']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Function to handle missing values and clean up actor/producer/genre columns
def clean_and_extract_columns(df):
    # Combine all genres into a list and remove NaN values
    genre_columns = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']
    df['genres'] = df[genre_columns].apply(lambda x: set(x.dropna()), axis=1)
    
    # Combine all actors into a list and remove NaN values
    actor_columns = ['actor_1', 'actor_2', 'actor_3', 'actor_4', 'actor_5', 'actor_6']
    df['actors'] = df[actor_columns].apply(lambda x: set(x.dropna()), axis=1)
    
    # Combine all producers into a list and remove NaN values
    producer_columns = ['producer_1', 'producer_2', 'producer_3']
    df['producers'] = df[producer_columns].apply(lambda x: set(x.dropna()), axis=1)
    
    # Replace inplace=True with assignment to the DataFrame
    df['budget'] = df['budget'].fillna(0)
    df['revenue'] = df['revenue'].fillna(0)

    return df

# Clean up the dataframe
df = clean_and_extract_columns(df)

# Preprocess Genres (Multi-label binarization)
mlb = MultiLabelBinarizer()
genres_matrix = mlb.fit_transform(df['genres'])

# Preprocess Actors (Multi-label binarization)
actors_matrix = mlb.fit_transform(df['actors'])

# Preprocess Producers (Multi-label binarization)
producers_matrix = mlb.fit_transform(df['producers'])

# Preprocess Overview with TF-IDF (Textual Features)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
overview_matrix = vectorizer.fit_transform(df['overview'].fillna(''))

# Standardize numerical features (vote_average, popularity, budget, revenue)
numerical_columns = ['vote_average', 'popularity', 'budget', 'revenue']
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(df[numerical_columns].fillna(0))

# Combine all features
combined_features = np.hstack([overview_matrix.toarray(), genres_matrix, actors_matrix, producers_matrix, numerical_data_scaled])

# Apply PCA (Principal Component Analysis) to reduce dimensionality and improve similarity calculation
pca = PCA(n_components=100)  # Reduce to 100 dimensions
reduced_features = pca.fit_transform(combined_features)

# Compute Cosine Similarity between movies
similarity_matrix = cosine_similarity(reduced_features)

# Save similarity matrix for later use (optional)
np.savetxt("movie_similarity_matrix.csv", similarity_matrix, delimiter=",")

# Function to get the top N most similar movies to a given movie
def get_similar_movies(movie_index, top_n=5):
    # Get the similarity scores for the given movie
    similarity_scores = similarity_matrix[movie_index]
    
    # Sort the movies based on similarity (excluding the movie itself)
    similar_movie_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    
    # Get the titles of the top N similar movies
    similar_movie_titles = df.iloc[similar_movie_indices]['title'].values
    return similar_movie_titles

# Command-line interaction to ask user for a movie and recommend similar movies
def recommend_similar_movies():
    # Display a list of movie titles
    print("Available movies:")
    for i, title in enumerate(df['title'].head(20)):  # Display first 20 movie titles
        print(f"{i}: {title}")
    
    # Ask user to input the index of the movie
    try:
        movie_index = int(input("Enter the index number of the movie you want to find similar films to: "))
        
        # Get top 5 similar movies
        recommended_titles = get_similar_movies(movie_index, top_n=5)
        
        # Print the recommended similar movies
        print(f"\nMovies similar to '{df.iloc[movie_index]['title']}':")
        for title in recommended_titles:
            print(title)
    except (ValueError, IndexError):
        print("Invalid input. Please make sure you select a valid movie index.")

# Run the recommend_similar_movies function to allow user interaction
recommend_similar_movies()
