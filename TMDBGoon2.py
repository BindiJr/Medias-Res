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
fieldnames = ["id", "title", "release_date", "genres", "runtime", "vote_average", "vote_count", "director", "cast", "overview"]

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
            runtime = movie_details.get("runtime", "N/A")
            vote_avg = movie_details.get("vote_average", 0)
            vote_count = movie_details.get("vote_count", 0)
            overview = movie_details.get("overview", "").replace("\n", " ").strip()

            # Get director (from crew data)
            director = "N/A"
            for crew_member in movie_details.get("credits", {}).get("crew", []):
                if crew_member["job"] == "Director":
                    director = crew_member["name"]
                    break

            # Get top 3 cast members
            cast = ", ".join([actor["name"] for actor in movie_details.get("credits", {}).get("cast", [])[:3]])

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
                "cast": cast.encode("utf-8", "ignore").decode("utf-8"),
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
