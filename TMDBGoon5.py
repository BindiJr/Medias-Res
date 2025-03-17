import requests
import csv
import time

API_KEY = "389ab8dcd1551542196f5e49238f57e2"  # Replace with your actual API key
BASE_URL = "https://api.themoviedb.org/3"

def get_movie_details(movie_id):
    """Fetch detailed movie info from TMDb."""
    url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US&append_to_response=credits"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for Movie ID {movie_id}: {response.status_code}")
        return None

def get_popular_movies(page=1):
    """Fetch popular movies from TMDb."""
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code}")
        return []

# Define CSV file
csv_file = "tmdb_movies.csv"
fieldnames = [
    "id", "title", "release_date", "genres", "runtime", "vote_average", "vote_count", 
    "director", "dop", "composer_1", "composer_2", "producer_1", "producer_2", "screenwriter_1", 
    "screenwriter_2", "editor_1", "editor_2", "actor_1", "actor_2", "actor_3", "actor_4", "actor_5", 
    "overview"
]

# Write headers to CSV (with UTF-8 encoding)
with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

# Fetch first 10 pages of movies (adjust as needed)
for page in range(1, 11):
    movies = get_popular_movies(page)
    movie_data = []

    for movie in movies:
        movie_details = get_movie_details(movie["id"])
        if movie_details:
            genres = ", ".join([g["name"] for g in movie_details.get("genres", [])])
            runtime = movie_details.get("runtime", "")
            vote_avg = movie_details.get("vote_average", 0)
            vote_count = movie_details.get("vote_count", 0)
            overview = movie_details.get("overview", "").replace("\n", " ").strip()

            # Get director (from crew data)
            director = ""
            for crew_member in movie_details.get("credits", {}).get("crew", []):
                if crew_member["job"] == "Director":
                    director = crew_member["name"]
                    break

            # Get DOP (Director of Photography)
            dop = ""
            for crew_member in movie_details.get("credits", {}).get("crew", []):
                if crew_member["job"] == "Director of Photography":
                    dop = crew_member["name"]
                    break

            # Get composer(s)
            composer_1, composer_2 = "", ""
            composers = [crew_member["name"] for crew_member in movie_details.get("credits", {}).get("crew", []) if crew_member["job"] == "Original Music Composer"]
            if composers:
                composer_1 = composers[0]
            if len(composers) > 1:
                composer_2 = composers[1]

            # Get producers
            producer_1, producer_2 = "", ""
            producers = [crew_member["name"] for crew_member in movie_details.get("credits", {}).get("crew", []) if crew_member["job"] == "Producer"]
            if producers:
                producer_1 = producers[0]
            if len(producers) > 1:
                producer_2 = producers[1]

            # Get screenwriters
            screenwriter_1, screenwriter_2 = "", ""
            screenwriters = [crew_member["name"] for crew_member in movie_details.get("credits", {}).get("crew", []) if crew_member["job"] == "Screenplay"]
            if screenwriters:
                screenwriter_1 = screenwriters[0]
            if len(screenwriters) > 1:
                screenwriter_2 = screenwriters[1]

            # Get editors
            editor_1, editor_2 = "", ""
            editors = [crew_member["name"] for crew_member in movie_details.get("credits", {}).get("crew", []) if crew_member["job"] == "Editor"]
            if editors:
                editor_1 = editors[0]
            if len(editors) > 1:
                editor_2 = editors[1]

            # Get top 5 cast members (split into separate columns)
            cast = movie_details.get("credits", {}).get("cast", [])[:5]
            actors = [actor["name"] for actor in cast]
            actor_1 = actors[0] if len(actors) > 0 else ""
            actor_2 = actors[1] if len(actors) > 1 else ""
            actor_3 = actors[2] if len(actors) > 2 else ""
            actor_4 = actors[3] if len(actors) > 3 else ""
            actor_5 = actors[4] if len(actors) > 4 else ""

            # Store data (ensure UTF-8 encoding)
            movie_data.append({
                "id": movie["id"],
                "title": movie["title"].encode("utf-8", "ignore").decode("utf-8"),
                "release_date": movie["release_date"],
                "genres": genres.encode("utf-8", "ignore").decode("utf-8"),
                "runtime": runtime,
                "vote_average": vote_avg,
                "vote_count": vote_count,
                "director": director.encode("utf-8", "ignore").decode("utf-8"),
                "dop": dop.encode("utf-8", "ignore").decode("utf-8"),
                "composer_1": composer_1.encode("utf-8", "ignore").decode("utf-8"),
                "composer_2": composer_2.encode("utf-8", "ignore").decode("utf-8"),
                "producer_1": producer_1.encode("utf-8", "ignore").decode("utf-8"),
                "producer_2": producer_2.encode("utf-8", "ignore").decode("utf-8"),
                "screenwriter_1": screenwriter_1.encode("utf-8", "ignore").decode("utf-8"),
                "screenwriter_2": screenwriter_2.encode("utf-8", "ignore").decode("utf-8"),
                "editor_1": editor_1.encode("utf-8", "ignore").decode("utf-8"),
                "editor_2": editor_2.encode("utf-8", "ignore").decode("utf-8"),
                "actor_1": actor_1.encode("utf-8", "ignore").decode("utf-8"),
                "actor_2": actor_2.encode("utf-8", "ignore").decode("utf-8"),
                "actor_3": actor_3.encode("utf-8", "ignore").decode("utf-8"),
                "actor_4": actor_4.encode("utf-8", "ignore").decode("utf-8"),
                "actor_5": actor_5.encode("utf-8", "ignore").decode("utf-8"),
                "overview": overview.encode("utf-8", "ignore").decode("utf-8")
            })

        # Be respectful of TMDb API limits
        time.sleep(0.5)

    # Write batch to CSV
    with open(csv_file, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(movie_data)

    print(f"Page {page} saved to CSV.")

print("All movies saved successfully!")
