import requests
import csv
import time

API_KEY = "389ab8dcd1551542196f5e49238f57e2"  # Replace with your actual API key
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(movie_id):
    """Fetch detailed movie info from TMDb."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=credits,release_dates,keywords"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for Movie ID {movie_id}: {response.status_code}")
        return None

def get_top_rated_movies(page=1):
    """Fetch top-rated movies from TMDb."""
    url = f"{BASE_URL}/movie/top_rated?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code}")
        return []

# Define CSV file
csv_file = "tmdb_top_rated_movies.csv"
fieldnames = [
    "id", "title", "release_date", "genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6",
    "runtime", "vote_average", "vote_count", "popularity", "budget", "revenue", "tagline",
    "director", "producer_1", "producer_2", "producer_3",
    "actor_1", "actor_2", "actor_3", "actor_4", "actor_5", "actor_6",
    "overview", "original_language", "homepage", "backdrop_path", "poster_path",
    "collection_name", "keywords", "adult_content"
]

# Write headers to CSV (with UTF-8 encoding)
with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Fetch top-rated movies sequentially
page = 1
while True:
    movies = get_top_rated_movies(page)
    if not movies:  # If no movies are returned, we are at the end of the list
        break

    movie_data = []
    
    for movie in movies:
        movie_details = get_movie_details(movie["id"])
        if movie_details:
            # Genres
            genres = movie_details.get("genres", [])
            genre_columns = [g["name"] for g in genres[:6]]  # Limit to 6 genres
            genre_columns += [""] * (6 - len(genre_columns))  # Fill with empty strings if less than 6

            # Other details
            runtime = movie_details.get("runtime", "")
            vote_avg = movie_details.get("vote_average", 0)
            vote_count = movie_details.get("vote_count", 0)
            popularity = movie_details.get("popularity", 0)
            budget = movie_details.get("budget", "")
            revenue = movie_details.get("revenue", "")
            tagline = movie_details.get("tagline", "")
            overview = movie_details.get("overview", "").replace("\n", " ").strip()

            # Remove budget and revenue if they're 0
            if budget == 0:
                budget = ""
            if revenue == 0:
                revenue = ""

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

            # Languages
            languages = movie_details.get("spoken_languages", [])
            language_names = [language["name"] for language in languages]

            # Original Language
            original_language = movie_details.get("original_language", "")

            # Collection Info
            collection_name = ""
            if movie_details.get("belongs_to_collection"):
                collection_name = movie_details["belongs_to_collection"].get("name", "")

            # Keywords (Reverted to previous strategy)
            keywords = movie_details.get("keywords", {}).get("keywords", [])
            keyword_names = [keyword["name"] for keyword in keywords]

            # Adult Content
            adult_content = movie_details.get("adult", False)

            # Store data
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
                "original_language": original_language,
                "homepage": movie_details.get("homepage", ""),
                "backdrop_path": movie_details.get("backdrop_path", ""),
                "poster_path": movie_details.get("poster_path", ""),
                "collection_name": collection_name.encode("utf-8", "ignore").decode("utf-8"),
                "keywords": ", ".join(keyword_names),
                "adult_content": adult_content
            })

        # Be respectful of TMDb API limits
        time.sleep(0.5)

    # Write batch to CSV
    with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(movie_data)

    print(f"Page {page} saved to CSV.")
    page += 1  # Move to the next page

print("All top-rated movies saved successfully!")
