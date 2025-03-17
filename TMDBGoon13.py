import requests
import csv
import time

API_KEY = "389ab8dcd1551542196f5e49238f57e2"  # Replace with your actual API key
BASE_URL = "https://api.themoviedb.org/3"
CSV_FILE = "tmdb_top_rated_movies.csv"
PAGE_SIZE = 50  # Fetch 50 movies per page

def get_top_rated_movies(page=1):
    """Fetch top-rated movies from TMDb."""
    url = f"{BASE_URL}/movie/top_rated?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code} while fetching top-rated movies from page {page}")
        return []

def get_movie_details(movie_id):
    """Fetch detailed movie info from TMDb."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=credits,release_dates,keywords"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for Movie ID {movie_id}: {response.status_code}")
        return None

def get_processed_movie_titles():
    """Get the list of already processed movie titles from the CSV file."""
    processed_titles = set()
    try:
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                processed_titles.add(row["title"].strip().lower())  # Track titles in lowercase
    except FileNotFoundError:
        return processed_titles  # Return empty set if CSV doesn't exist yet
    return processed_titles

def write_movie_data_to_csv(movie_data):
    """Write the movie data to CSV."""
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=movie_data[0].keys())
        writer.writerows(movie_data)

def process_movies():
    """Main function to process movies."""
    processed_titles = get_processed_movie_titles()  # Get the list of processed movies
    current_page = 1
    total_pages = 50  # You can adjust this as needed
    fetched_movies = 0  # Track the number of movies processed

    # Write headers if CSV is empty
    if not processed_titles:
        fieldnames = [
            "id", "title", "release_date", "genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6",
            "runtime", "vote_average", "vote_count", "popularity", "budget", "revenue", "tagline",
            "director", "producer_1", "producer_2", "producer_3",
            "actor_1", "actor_2", "actor_3", "actor_4", "actor_5", "actor_6",
            "overview", "original_language", "homepage", "backdrop_path", "poster_path",
            "collection_name", "keywords", "adult_content"
        ]
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    # Process movies in chunks of 50 pages
    while True:
        print(f"Processing page {current_page}...")
        movies = get_top_rated_movies(current_page)
        
        if not movies:  # No more movies to process
            print(f"End of movies reached at page {current_page}")
            break

        movie_data = []
        for movie in movies:
            movie_title = movie["title"].strip().lower()

            # Skip if the movie has already been processed
            if movie_title in processed_titles:
                print(f"Skipping movie {movie_title} (already processed)")
                continue

            movie_details = get_movie_details(movie["id"])
            if movie_details:
                # Extract movie details as in the original script
                genres = movie_details.get("genres", [])
                genre_columns = [g["name"] for g in genres[:6]]  # Limit to 6 genres
                genre_columns += [""] * (6 - len(genre_columns))  # Fill with empty strings if less than 6

                # Other details like runtime, vote average, etc.
                runtime = movie_details.get("runtime", "")
                vote_avg = movie_details.get("vote_average", 0)
                vote_count = movie_details.get("vote_count", 0)
                popularity = movie_details.get("popularity", 0)
                budget = movie_details.get("budget", "")
                revenue = movie_details.get("revenue", "")
                tagline = movie_details.get("tagline", "")
                overview = movie_details.get("overview", "").replace("\n", " ").strip()

                # Director and Producers
                director = ""
                producers = []
                for crew_member in movie_details.get("credits", {}).get("crew", []):
                    if crew_member["job"] == "Director":
                        director = crew_member["name"]
                    if crew_member["job"] == "Producer" and len(producers) < 3:
                        producers.append(crew_member["name"])

                # Top 6 Actors
                cast = movie_details.get("credits", {}).get("cast", [])[:6]
                actors = [actor["name"] for actor in cast]
                actor_columns = actors + [""] * (6 - len(actors))  # Fill with empty strings if less than 6

                # Collection Info
                collection_name = ""
                if movie_details.get("belongs_to_collection"):
                    collection_name = movie_details["belongs_to_collection"].get("name", "")

                # Keywords
                keywords = movie_details.get("keywords", {}).get("keywords", [])
                keyword_names = [keyword["name"] for keyword in keywords]

                # Adult Content
                adult_content = movie_details.get("adult", False)

                # Store movie data
                movie_data.append({
                    "id": movie["id"],
                    "title": movie["title"].encode("utf-8", "ignore").decode("utf-8"),
                    "release_date": movie["release_date"],
                    "genre_1": genre_columns[0], "genre_2": genre_columns[1], "genre_3": genre_columns[2],
                    "genre_4": genre_columns[3], "genre_5": genre_columns[4], "genre_6": genre_columns[5],
                    "runtime": runtime,
                    "vote_average": vote_avg,
                    "vote_count": vote_count,
                    "popularity": popularity,
                    "budget": budget,
                    "revenue": revenue,
                    "tagline": tagline.encode("utf-8", "ignore").decode("utf-8"),
                    "director": director.encode("utf-8", "ignore").decode("utf-8"),
                    "producer_1": producers[0] if len(producers) > 0 else "",
                    "producer_2": producers[1] if len(producers) > 1 else "",
                    "producer_3": producers[2] if len(producers) > 2 else "",
                    "actor_1": actor_columns[0].encode("utf-8", "ignore").decode("utf-8"),
                    "actor_2": actor_columns[1].encode("utf-8", "ignore").decode("utf-8"),
                    "actor_3": actor_columns[2].encode("utf-8", "ignore").decode("utf-8"),
                    "actor_4": actor_columns[3].encode("utf-8", "ignore").decode("utf-8"),
                    "actor_5": actor_columns[4].encode("utf-8", "ignore").decode("utf-8"),
                    "actor_6": actor_columns[5].encode("utf-8", "ignore").decode("utf-8"),
                    "overview": overview.encode("utf-8", "ignore").decode("utf-8"),
                    "original_language": movie_details.get("original_language", ""),
                    "homepage": movie_details.get("homepage", ""),
                    "backdrop_path": movie_details.get("backdrop_path", ""),
                    "poster_path": movie_details.get("poster_path", ""),
                    "collection_name": collection_name.encode("utf-8", "ignore").decode("utf-8"),
                    "keywords": ", ".join(keyword_names),
                    "adult_content": adult_content
                })

                processed_titles.add(movie_title)  # Add to processed titles set
                fetched_movies += 1

        if movie_data:
            write_movie_data_to_csv(movie_data)
            print(f"Page {current_page} processed. Last processed movie: {movie_data[-1]['title']}")

        # Be respectful of TMDb API limits
        time.sleep(1)  # Sleep for 1 second between requests to avoid rate limits

        current_page += 1  # Go to next page

    print(f"Total movies fetched and saved: {fetched_movies}")

# Start processing
process_movies()
