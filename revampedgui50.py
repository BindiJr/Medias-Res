import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QFormLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QGridLayout)
from PySide6.QtCore import Qt
from PySide6.QtGui import (QFont, QShortcut, QKeySequence)
import pandas as pd
import time
import random
import numpy as np
from collections import Counter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        return super(NumpyEncoder, self).default(obj)

def save_similarity_matrix(similarity_matrix, file_name="similarity_matrix.npy"):
    file_path = os.path.join(os.getcwd(), file_name)
    np.save(file_path, similarity_matrix)
    print(f"Similarity matrix saved to {file_path}")

def load_similarity_matrix(file_name="similarity_matrix.npy"):
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        print("Loading pre-computed similarity matrix...")
        return np.load(file_path)
    return None

def preprocess_movie_data(df):
    # Combine genre columns into a single list
    genre_columns = ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'genre_5', 'genre_6']
    df['genres'] = df[genre_columns].apply(lambda x: [g for g in x if pd.notna(g)], axis=1)
    return df

class FilmRecommendationScreen(QMainWindow):
    def __init__(self, username, user_profile, similarity_matrix, movie_data):
        super().__init__()
        self.username = username
        self.user_profile = user_profile
        self.similarity_matrix = similarity_matrix
        self.movie_data = movie_data
        
        # Initialize recommendation queue with a small batch
        self.recommendation_queue = []
        self.batch_size = 10  # Load recommendations in smaller batches
        self.all_shown_recommendations = set()  # Track all shown recommendations
        self.current_index = 0
        self.total_recommendations_shown = 0
        
        # Track interaction history for learning
        if "interaction_history" not in self.user_profile:
            self.user_profile["interaction_history"] = []
        
        # Initialize UI
        self.setup_ui()
        
        # Show first recommendation
        self.refresh_recommendation_queue()
        self.show_current_recommendation()
    
    def setup_ui(self):
        self.setWindowTitle("Film Recommendations")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Film display
        self.film_display = QVBoxLayout()
        
        self.film_title = QLabel("")
        self.film_title.setAlignment(Qt.AlignCenter)
        self.film_title.setFont(QFont("Arial", 18, QFont.Bold))
        self.film_display.addWidget(self.film_title)
        
        self.film_details = QLabel("")
        self.film_details.setAlignment(Qt.AlignCenter)
        self.film_details.setWordWrap(True)
        self.film_display.addWidget(self.film_details)
        
        main_layout.addLayout(self.film_display)
        
        # Replace the horizontal buttons_layout with a grid layout
        buttons_layout = QGridLayout()

        # Reset recommendations button (up position, like number 8)
        self.reset_button = QPushButton("Reset Recommendations")
        self.reset_button.setFixedSize(150, 50)
        self.reset_button.clicked.connect(self.reset_recommendations)
        buttons_layout.addWidget(self.reset_button, 0, 1)  # Row 0, Column 1 (top middle)

        # Not interested button (left position, like number 4)
        self.not_interested_button = QPushButton("Not Interested")
        self.not_interested_button.setFixedSize(150, 50)
        self.not_interested_button.clicked.connect(self.swipe_left)
        buttons_layout.addWidget(self.not_interested_button, 1, 0)  # Row 1, Column 0 (middle left)

        # Already seen button (down position, like number 5)
        self.already_seen_button = QPushButton("Already Seen")
        self.already_seen_button.setFixedSize(150, 50)
        self.already_seen_button.clicked.connect(self.mark_as_seen)
        buttons_layout.addWidget(self.already_seen_button, 1, 1)  # Row 1, Column 1 (middle)

        # Want to watch button (right position, like number 6)
        self.want_to_watch_button = QPushButton("Want to Watch")
        self.want_to_watch_button.setFixedSize(150, 50)
        self.want_to_watch_button.clicked.connect(self.swipe_right)
        buttons_layout.addWidget(self.want_to_watch_button, 1, 2)  # Row 1, Column 2 (middle right)

        main_layout.addLayout(buttons_layout)
        
        # Progress indicator
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_label)
        
        # Set up keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.swipe_left)
        
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.swipe_right)
        
        self.shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut_up.activated.connect(self.reset_recommendations)
        
        self.shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut_down.activated.connect(self.mark_as_seen)
    
    def refresh_recommendation_queue(self):
        """Add a new batch of recommendations to the queue"""
        # Only generate new recommendations if queue is getting low
        if len(self.recommendation_queue) < self.batch_size:
            # Get a larger batch to allow for filtering
            new_batch = self.generate_recommendations(self.batch_size * 3)  # Get more than needed
            
            # Filter out any films already shown
            filtered_batch = [film_id for film_id in new_batch 
                            if film_id not in self.all_shown_recommendations]
            
            # If we're running out of new recommendations, allow some repeats but prioritize less recently shown ones
            if len(filtered_batch) < self.batch_size and len(new_batch) >= self.batch_size:
                print("Running low on new recommendations, allowing some repeats")
                # Add some films that were shown longest ago
                recent_recommendations = list(self.user_profile.get("interaction_history", []))
                recent_film_ids = [int(item["film_id"]) for item in recent_recommendations if "film_id" in item]
                
                # Add films from new_batch that weren't recently interacted with
                for film_id in new_batch:
                    if film_id not in filtered_batch and film_id not in recent_film_ids[-20:]:
                        filtered_batch.append(film_id)
                        if len(filtered_batch) >= self.batch_size:
                            break
            
            # Add to queue
            self.recommendation_queue.extend(filtered_batch)
            
            # Add these films to the shown recommendations set
            self.all_shown_recommendations.update(filtered_batch)
            
            print(f"Queue refreshed, now has {len(self.recommendation_queue)} recommendations")
    
    def generate_recommendations(self, batch_size=10):
        """Simplified recommendation pipeline"""
        # 1. Gather exclusion list (films to exclude from recommendations)
        excluded_films = self.get_excluded_films()
        
        # 2. Generate candidate pool from multiple sources
        candidates = []
        
        # 2a. Get similarity-based candidates
        similarity_candidates = self.get_similarity_candidates(excluded_films)
        candidates.extend(similarity_candidates)
        
        # 2b. Get genre-based candidates if needed
        if len(candidates) < batch_size * 2:
            genre_candidates = self.get_genre_candidates(excluded_films)
            candidates.extend([c for c in genre_candidates if c not in candidates])
        
        # 2c. Get popularity-based candidates if needed
        if len(candidates) < batch_size:
            popularity_candidates = self.get_popularity_candidates(excluded_films)
            candidates.extend([c for c in popularity_candidates if c not in candidates])
        
        # 3. Score and rank candidates
        scored_candidates = self.score_candidates(candidates)
        
        # 4. Add diversity through randomized selection
        if len(scored_candidates) > batch_size:
            # Use weighted random sampling favoring higher scores
            weights = [score for _, score in scored_candidates]
            selected_indices = random.choices(
                range(len(scored_candidates)), 
                weights=weights, 
                k=min(batch_size, len(scored_candidates))
            )
            final_recommendations = [scored_candidates[i][0] for i in selected_indices]
        else:
            final_recommendations = [c[0] for c in scored_candidates]
        
        return final_recommendations
    
    def get_excluded_films(self):
        """Get set of films to exclude from recommendations"""
        seen_films = set(self.user_profile.get("seen_movies", {}).keys())
        not_interested = set(self.user_profile.get("not_interested", []))
        want_to_watch = set(self.user_profile.get("want_to_watch", {}).keys())
        
        return seen_films.union(not_interested).union(want_to_watch)
    
    def get_similarity_candidates(self, excluded_films, max_candidates=30):
        """Get candidates based on similarity to highly rated films"""
        candidates = []
        
        # Get seed films (highly rated or recently liked)
        seed_films = self.get_seed_films(excluded_films)
        
        # If no seed films, return empty list
        if not seed_films:
            return []
        
        # Find similar films for each seed film
        for film_id in seed_films:
            if film_id >= len(self.similarity_matrix):
                continue
                
            # Get similar films
            similar_films = np.argsort(self.similarity_matrix[film_id])[::-1][:20]
            
            # Add to candidates if not excluded
            for similar_id in similar_films:
                if str(similar_id) not in excluded_films and similar_id not in candidates:
                    candidates.append(similar_id)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def get_seed_films(self, excluded_films, max_seeds=10):
        """Get seed films for recommendation generation"""
        seed_candidates = []
        
        # Add highly rated films
        if "film_ratings" in self.user_profile and self.user_profile["film_ratings"]:
            ratings = self.user_profile["film_ratings"]
            threshold = np.percentile(list(ratings.values()), 75)
            favorite_films = [int(film_id) for film_id, rating in ratings.items() 
                             if rating >= threshold]
            seed_candidates.extend(favorite_films)
        
        # Add films from recent positive interactions
        recent_interactions = self.user_profile.get("interaction_history", [])[-20:]
        liked_films = [int(item["film_id"]) for item in recent_interactions 
                      if item["action"] == "want_to_watch"]
        seed_candidates.extend(liked_films)
        
        # Remove duplicates and excluded films
        seed_candidates = [f for f in seed_candidates if str(f) not in excluded_films]
        
        # Add randomness to seed selection
        if len(seed_candidates) > max_seeds:
            return random.sample(seed_candidates, max_seeds)
        
        return seed_candidates
    
    def get_genre_candidates(self, excluded_films, max_candidates=20):
        """Get candidates based on genre preferences"""
        # Get liked genres
        liked_genres = [genre for genre, rating in 
                       self.user_profile.get("genre_preferences", {}).items() 
                       if rating == 1]
        
        if not liked_genres:
            return []
        
        # Get popular films from liked genres
        candidates = []
        for genre in liked_genres:
            # Filter movies where this genre is the primary genre
            films = self.movie_data[self.movie_data['genre_1'] == genre]
            # Sort by popularity
            films = films.sort_values('popularity', ascending=False)
            # Add to candidates if not excluded
            for film_id in films.index:
                if str(film_id) not in excluded_films and film_id not in candidates:
                    candidates.append(film_id)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def get_popularity_candidates(self, excluded_films, max_candidates=10):
        """Get candidates based on popularity"""
        candidates = []
        popular_movies = self.movie_data.sort_values('popularity', ascending=False)
        
        for film_id in popular_movies.index:
            if str(film_id) not in excluded_films and film_id not in candidates:
                candidates.append(film_id)
                if len(candidates) >= max_candidates:
                    break
        
        return candidates
    
    def score_candidates(self, candidates):
        """Score and rank candidate films"""
        scores = []
        
        # Get seed films for similarity comparison
        seed_films = self.get_seed_films(set(), max_seeds=5)
        
        for film_id in candidates:
            # Calculate similarity score
            similarity_score = 0
            for seed_id in seed_films:
                if seed_id < len(self.similarity_matrix) and film_id < len(self.similarity_matrix):
                    similarity_score += self.similarity_matrix[seed_id][film_id]
            
            # Normalize by number of seed films
            if seed_films:
                similarity_score /= len(seed_films)
            
            # Add to scores
            scores.append((film_id, similarity_score))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def show_current_recommendation(self):
        # Check if we need to refresh the queue
        if self.current_index >= len(self.recommendation_queue):
            self.refresh_recommendation_queue()
            self.current_index = 0
        
        # If still no recommendations, show a message
        if not self.recommendation_queue or self.current_index >= len(self.recommendation_queue):
            QMessageBox.information(self, "No Recommendations", 
                                   "No film recommendations available. Try resetting recommendations.")
            return
        
        # Get current film
        film_id = self.recommendation_queue[self.current_index]
        film = self.movie_data.loc[film_id]
        
        # Update display
        self.film_title.setText(film['title'])
        
        # Create details text with release year, director, etc.
        details = f"Released: {film['release_date'][:4]}\n"
        details += f"Director: {film['director']}\n"
        
        # Safely handle genres
        try:
            genres = film.get('genres', [])
            if isinstance(genres, list):
                genres_text = ', '.join(genres)
            else:
                genres_text = 'N/A'
        except:
            genres_text = 'N/A'
        
        details += f"Genres: {genres_text}\n"
        details += f"Overview: {film['overview'][:300]}..." if len(film['overview']) > 300 else f"Overview: {film['overview']}"
        
        self.film_details.setText(details)
        
        # Track total recommendations shown (not just current batch)
        self.total_recommendations_shown += 1
        
        # Update progress with continuous counter
        self.progress_label.setText(f"Recommendation #{self.total_recommendations_shown}")

    def swipe_left(self):
        """User is not interested in this film"""
        if self.current_index >= len(self.recommendation_queue):
            return
            
        film_id = self.recommendation_queue[self.current_index]
        
        # Record not interested
        if "not_interested" not in self.user_profile:
            self.user_profile["not_interested"] = []
        
        self.user_profile["not_interested"].append(str(film_id))
        
        # Add to interaction history
        self.user_profile["interaction_history"].append({
            "film_id": str(film_id),
            "action": "not_interested",
            "timestamp": time.time()
        })
        
        # Move to next recommendation
        self.current_index += 1
        self.save_user_profile()
        self.show_current_recommendation()

    def swipe_right(self):
        """User wants to watch this film"""
        if self.current_index >= len(self.recommendation_queue):
            return
            
        film_id = self.recommendation_queue[self.current_index]
        
        # Add to want_to_watch list
        if "want_to_watch" not in self.user_profile:
            self.user_profile["want_to_watch"] = {}
        
        self.user_profile["want_to_watch"][str(film_id)] = {
            "title": self.movie_data.loc[film_id]['title'],
            "added_at": time.time()
        }
        
        # Add to interaction history
        self.user_profile["interaction_history"].append({
            "film_id": str(film_id),
            "action": "want_to_watch",
            "timestamp": time.time()
        })
        
        # Save profile and move to next recommendation
        self.current_index += 1
        self.save_user_profile()
        
        # Refresh recommendations to incorporate this preference
        # This creates the learning effect - new recommendations will be influenced
        # by this positive interaction
        if len(self.user_profile["interaction_history"]) % 5 == 0:
            # Refresh every 5 interactions to adapt to user preferences
            self.recommendation_queue = self.recommendation_queue[:self.current_index]
            self.refresh_recommendation_queue()
        
        self.show_current_recommendation()

    def mark_as_seen(self):
        """User has already seen this film"""
        if self.current_index >= len(self.recommendation_queue):
            return
            
        film_id = self.recommendation_queue[self.current_index]
        film = self.movie_data.loc[film_id]
        
        # Add to seen_movies
        if "seen_movies" not in self.user_profile:
            self.user_profile["seen_movies"] = {}
        
        # Get genres safely
        try:
            genres = film.get('genres', [])
            if not isinstance(genres, list):
                genres = [film.get('genre_1', '')]
        except:
            genres = [film.get('genre_1', '')]
        
        self.user_profile["seen_movies"][str(film_id)] = {
            "title": film['title'],
            "genres": genres,
            "rating": None,
            "timestamp": time.time()
        }
        
        # Add to interaction history
        self.user_profile["interaction_history"].append({
            "film_id": str(film_id),
            "action": "already_seen",
            "timestamp": time.time()
        })
        
        # Move to next recommendation
        self.current_index += 1
        self.save_user_profile()
        self.show_current_recommendation()

    def reset_recommendations(self):
        """Reset recommendations when they're off-base"""
        # Clear the queue and generate fresh recommendations
        self.recommendation_queue = []
        self.refresh_recommendation_queue()
        self.current_index = 0
        
        # Record this reset in interaction history
        self.user_profile["interaction_history"].append({
            "action": "reset_recommendations",
            "timestamp": time.time()
        })
        
        self.save_user_profile()
        self.show_current_recommendation()

    def save_user_profile(self):
        profile_dir = os.path.join(os.getcwd(), "user_profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{self.username}.json")
        with open(profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=4, cls=NumpyEncoder)
        print(f"User profile saved to {profile_path}")

    def get_excluded_films(self):
        """Get set of films to exclude from recommendations"""
        seen_films = set(self.user_profile.get("seen_movies", {}).keys())
        not_interested = set(self.user_profile.get("not_interested", []))
        want_to_watch = set(self.user_profile.get("want_to_watch", {}).keys())
        
        return seen_films.union(not_interested).union(want_to_watch)

    def get_similarity_candidates(self, excluded_films, max_candidates=30):
        """Get candidates based on similarity to highly rated films"""
        candidates = []
        
        # Get seed films (highly rated or recently liked)
        seed_films = self.get_seed_films(excluded_films)
        
        # If no seed films, return empty list
        if not seed_films:
            return []
        
        # Find similar films for each seed film
        for film_id in seed_films:
            if film_id >= len(self.similarity_matrix):
                continue
                
            # Get similar films
            similar_films = np.argsort(self.similarity_matrix[film_id])[::-1][:20]
            
            # Add to candidates if not excluded
            for similar_id in similar_films:
                if str(similar_id) not in excluded_films and similar_id not in candidates:
                    candidates.append(similar_id)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates

    def get_seed_films(self, excluded_films, max_seeds=10):
        """Get seed films for recommendation generation"""
        seed_candidates = []
        
        # Add highly rated films
        if "film_ratings" in self.user_profile and self.user_profile["film_ratings"]:
            ratings = self.user_profile["film_ratings"]
            threshold = np.percentile(list(ratings.values()), 75)
            favorite_films = [int(film_id) for film_id, rating in ratings.items() 
                            if rating >= threshold]
            seed_candidates.extend(favorite_films)
        
        # Add films from recent positive interactions
        recent_interactions = self.user_profile.get("interaction_history", [])[-20:]
        liked_films = [int(item["film_id"]) for item in recent_interactions 
                    if item["action"] == "want_to_watch"]
        seed_candidates.extend(liked_films)
        
        # Remove duplicates and excluded films
        seed_candidates = [f for f in seed_candidates if str(f) not in excluded_films]
        
        # Add randomness to seed selection
        if len(seed_candidates) > max_seeds:
            return random.sample(seed_candidates, max_seeds)
        
        return seed_candidates

    def get_genre_candidates(self, excluded_films, max_candidates=20):
        """Get candidates based on genre preferences"""
        # Get liked genres
        liked_genres = [genre for genre, rating in 
                    self.user_profile.get("genre_preferences", {}).items() 
                    if rating == 1]
        
        if not liked_genres:
            return []
        
        # Get popular films from liked genres
        candidates = []
        for genre in liked_genres:
            # Filter movies where this genre is the primary genre
            films = self.movie_data[self.movie_data['genre_1'] == genre]
            # Sort by popularity
            films = films.sort_values('popularity', ascending=False)
            # Add to candidates if not excluded
            for film_id in films.index:
                if str(film_id) not in excluded_films and film_id not in candidates:
                    candidates.append(film_id)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates

    def get_popularity_candidates(self, excluded_films, max_candidates=10):
        """Get candidates based on popularity"""
        candidates = []
        popular_movies = self.movie_data.sort_values('popularity', ascending=False)
        
        for film_id in popular_movies.index:
            if str(film_id) not in excluded_films and film_id not in candidates:
                candidates.append(film_id)
                if len(candidates) >= max_candidates:
                    break
        
        return candidates

    def score_candidates(self, candidates):
        """Score and rank candidate films"""
        scores = []
        
        # Get seed films for similarity comparison
        seed_films = self.get_seed_films(set(), max_seeds=5)
        
        for film_id in candidates:
            # Calculate similarity score
            similarity_score = 0
            for seed_id in seed_films:
                if seed_id < len(self.similarity_matrix) and film_id < len(self.similarity_matrix):
                    similarity_score += self.similarity_matrix[seed_id][film_id]
            
            # Normalize by number of seed films
            if seed_films:
                similarity_score /= len(seed_films)
            
            # Add to scores
            scores.append((film_id, similarity_score))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)

class FilmComparisonScreen(QMainWindow):
    def __init__(self, username, user_profile, similarity_matrix, movie_data):
        super().__init__()
        self.username = username
        self.user_profile = user_profile
        self.similarity_matrix = similarity_matrix
        self.movie_data = movie_data
        self.seen_movies = list(self.user_profile.get("seen_movies", {}).keys())
        
        if len(self.seen_movies) < 2:
            QMessageBox.information(self, "Not Enough Movies", 
                                "You need to have seen at least 2 movies to compare them.")
            self.close()
            return
            
        # Initialize comparison tracking
        self.null_film_id = self.select_initial_null_film()
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.comparison_count = 0
        self.max_comparisons = min(40, len(self.seen_movies) * 3)
        self.compared_pairs = set()  # Track which pairs have been compared
        self.film_win_counts = Counter()  # Track wins for each film
        self.null_film_counter = 0  # Add this line to initialize the counter
        
        # Setup UI components
        self.setup_ui()
        
        # Start first comparison
        self.show_next_comparison()
    
    def setup_ui(self):
        self.setWindowTitle("Compare Films")
        self.setGeometry(100, 100, 1000, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Instructions
        instructions = QLabel("Compare these films - which do you prefer?")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 14))
        main_layout.addWidget(instructions)
        
        # Films comparison area (without preference buttons)
        comparison_layout = QHBoxLayout()
        
        # Left film (null)
        self.left_panel = QVBoxLayout()
        self.left_title = QLabel("")
        self.left_title.setAlignment(Qt.AlignCenter)
        self.left_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.left_details = QLabel("")
        self.left_details.setWordWrap(True)
        self.left_panel.addWidget(self.left_title)
        self.left_panel.addWidget(self.left_details)
        
        # Right film (alternative)
        self.right_panel = QVBoxLayout()
        self.right_title = QLabel("")
        self.right_title.setAlignment(Qt.AlignCenter)
        self.right_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.right_details = QLabel("")
        self.right_details.setWordWrap(True)
        self.right_panel.addWidget(self.right_title)
        self.right_panel.addWidget(self.right_details)
        
        comparison_layout.addLayout(self.left_panel)
        comparison_layout.addLayout(self.right_panel)
        main_layout.addLayout(comparison_layout)
        
        # "That is enough comparison" button directly above preference buttons
        finish_button = QPushButton("That is enough comparison")
        finish_button.clicked.connect(self.finish_comparisons)
        # Remove the setFixedHeight line to match other buttons' height

        # Create a vertical layout for the middle column (finish + equal buttons)
        middle_column = QVBoxLayout()
        middle_column.addWidget(finish_button)

        # Preference buttons in a horizontal row
        preference_layout = QHBoxLayout()

        # Left preference button
        left_button = QPushButton("I prefer this film")
        left_button.clicked.connect(lambda: self.record_preference("left"))
        preference_layout.addWidget(left_button)

        # Equal button in the middle
        equal_button = QPushButton("These films are equal")
        equal_button.clicked.connect(lambda: self.record_preference("equal"))
        # Add equal button to the middle column instead of directly to preference_layout
        middle_column.addWidget(equal_button)

        # Add the middle column to the preference layout
        preference_layout.addLayout(middle_column)

        # Right preference button
        right_button = QPushButton("I prefer this film")
        right_button.clicked.connect(lambda: self.record_preference("right"))
        preference_layout.addWidget(right_button)

        main_layout.addLayout(preference_layout)
        
        # Progress indicator
        self.progress_label = QLabel(f"Comparison 1 of 40")
        self.progress_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_label)
        
        # Set up keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.record_preference("left"))
        
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.record_preference("right"))
        
        self.shortcut_down = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.shortcut_down.activated.connect(lambda: self.record_preference("equal"))
        
        self.shortcut_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.shortcut_up.activated.connect(self.finish_comparisons)
    
    def select_initial_null_film(self):
        """Select a diverse initial film to start comparisons"""
        if "genre_preferences" in self.user_profile:
            liked_genres = [g for g, p in self.user_profile["genre_preferences"].items() if p > 0]
            if liked_genres:
                genre_films = [m for m in self.seen_movies 
                              if any(g in liked_genres for g in 
                                    self.user_profile["seen_movies"][m].get("genres", []))]
                if genre_films:
                    return random.choice(genre_films)
        
        return random.choice(self.seen_movies)
    
    def select_next_null_film(self):
        """Intelligently select the next null film based on current preferences"""
        comparison_counts = Counter()
        for comp in self.user_profile.get("film_comparisons", []):
            comparison_counts[comp["film1_id"]] += 1
            comparison_counts[comp["film2_id"]] += 1
        
        avg_comparisons = sum(comparison_counts.values()) / max(1, len(comparison_counts))
        underrepresented = [m for m in self.seen_movies 
                           if comparison_counts[m] < avg_comparisons * 0.7]
        
        if underrepresented and random.random() < 0.4:
            return random.choice(underrepresented)
        
        if "film_ratings" in self.user_profile and len(self.user_profile["film_ratings"]) > 3:
            ratings = [(m, r) for m, r in self.user_profile["film_ratings"].items() 
                      if m in self.seen_movies]
            
            if ratings:
                ratings.sort(key=lambda x: x[1])  # Sort by rating value
                start_idx = len(ratings) // 3
                end_idx = start_idx * 2
                middle_third = ratings[start_idx:end_idx]
                if middle_third:
                    return random.choice(middle_third)[0]  # Return movie ID
        
        if "genre_preferences" in self.user_profile:
            neutral_genres = [g for g, p in self.user_profile["genre_preferences"].items() if p == 0]
            if neutral_genres:
                neutral_films = [m for m in self.seen_movies 
                               if any(g in neutral_genres for g in 
                                     self.user_profile["seen_movies"][m].get("genres", []))]
                if neutral_films:
                    return random.choice(neutral_films)
        
        recent_nulls = [comp["film1_id"] for comp in self.user_profile.get("film_comparisons", [])[-10:]]
        candidates = [m for m in self.seen_movies if m not in recent_nulls]
        if not candidates:
            candidates = self.seen_movies
        
        return random.choice(candidates)
    
    def select_alternative_film(self, null_film_id):
        """Select an appropriate alternative film to compare with the null film"""
        null_idx = int(null_film_id)
        
        potential_alts = [m for m in self.seen_movies if m != null_film_id]
        if not potential_alts:
            return None
            
        pair_key = lambda a, b: f"{min(a, b)}_{max(a, b)}"
        uncomp_pairs = [m for m in potential_alts 
                    if pair_key(null_film_id, m) not in self.compared_pairs]
        
        if uncomp_pairs and random.random() < 0.3:
            return random.choice(uncomp_pairs)
        
        if random.random() < 0.4:
            similarities = [(m, self.similarity_matrix[null_idx][int(m)]) 
                        for m in potential_alts]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            top_third = similarities[:max(1, len(similarities) // 3)]
            if top_third:
                return random.choice(top_third)[0]
        
        # Safely access genres and decade
        null_film_data = self.movie_data.loc[null_idx]
        
        # Check if 'genres' column exists, if not use empty set
        null_genres = set(null_film_data.get('genres', []))
        null_decade = null_film_data.get('decade', 0)
        
        different_films = []
        for alt_id in potential_alts:
            alt_idx = int(alt_id)
            alt_film_data = self.movie_data.loc[alt_idx]
            
            # Safely access genres and decade for alternative film
            alt_genres = set(alt_film_data.get('genres', []))
            alt_decade = alt_film_data.get('decade', 0)
            
            # Only calculate overlap if both have genres
            if null_genres and alt_genres:
                genre_overlap = len(null_genres.intersection(alt_genres)) / max(1, len(null_genres.union(alt_genres)))
            else:
                genre_overlap = 0
                
            decade_diff = abs(null_decade - alt_decade) / 10
            
            if genre_overlap < 0.5 or decade_diff >= 2:
                different_films.append(alt_id)
        
        if different_films:
            return random.choice(different_films)
        
        return random.choice(potential_alts)

    
    def show_next_comparison(self):
        # Check if we need a new null film
        if self.consecutive_wins >= 4 or self.consecutive_losses >= 4 or self.null_film_counter >= 8:
            if self.consecutive_losses >= 4:
                self.null_film_id = self.current_alt_film_id
            else:
                self.null_film_id = self.select_next_null_film()
            
            self.consecutive_wins = 0
            self.consecutive_losses = 0
            self.null_film_counter = 0  # Reset counter when changing null film
        
        # Get null film details
        null_film_data = self.movie_data.loc[int(self.null_film_id)]
        
        # Select alternative film
        alt_film_id = self.select_alternative_film(self.null_film_id)
        if alt_film_id is None:
            self.finish_comparisons()
            return
            
        alt_film_data = self.movie_data.loc[int(alt_film_id)]
        
        # Update UI with film details
        self.left_title.setText(null_film_data['title'])
        
        # Safely access genres
        try:
            null_genres = null_film_data.get('genres', [])
            if isinstance(null_genres, list):
                genres_text = ', '.join(null_genres)
            else:
                genres_text = 'N/A'
        except:
            genres_text = 'N/A'
        
        self.left_details.setText(f"Director: {null_film_data['director']}\n"
                                f"Released: {null_film_data['release_date'][:4]}\n"
                                f"Genres: {genres_text}")
        
        # Do the same for right panel
        self.right_title.setText(alt_film_data['title'])
        
        try:
            alt_genres = alt_film_data.get('genres', [])
            if isinstance(alt_genres, list):
                genres_text = ', '.join(alt_genres)
            else:
                genres_text = 'N/A'
        except:
            genres_text = 'N/A'
        
        self.right_details.setText(f"Director: {alt_film_data['director']}\n"
                                f"Released: {alt_film_data['release_date'][:4]}\n"
                                f"Genres: {genres_text}")
        
        # Store current comparison
        self.current_alt_film_id = alt_film_id
        
        # Update progress
        self.comparison_count += 1
        self.progress_label.setText(f"Comparison {self.comparison_count} of {self.max_comparisons}")
        
        # Add this pair to compared pairs
        pair_key = f"{min(self.null_film_id, alt_film_id)}_{max(self.null_film_id, alt_film_id)}"
        self.compared_pairs.add(pair_key)
    
    def record_preference(self, preference):
        # Initialize comparison data if not present
        if "film_comparisons" not in self.user_profile:
            self.user_profile["film_comparisons"] = []
        
        # Record the comparison result
        comparison = {
            "timestamp": time.time(),
            "film1_id": self.null_film_id,
            "film2_id": self.current_alt_film_id,
            "preference": preference
        }
        
        self.user_profile["film_comparisons"].append(comparison)
        
        # Update consecutive win/loss tracking
        if preference == "left":
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.film_win_counts[self.null_film_id] += 1
        elif preference == "right":
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.film_win_counts[self.current_alt_film_id] += 1
        else:  # equal
            self.consecutive_wins = 0
            self.consecutive_losses = 0
        
        # Update null film counter
        self.null_film_counter += 1

        # Save user profile
        self.save_user_profile()
        
        # Check if we've reached the maximum comparisons
        if self.comparison_count >= self.max_comparisons:
            self.finish_comparisons()
            return
        
        # Show next comparison
        self.show_next_comparison()
    
    def save_user_profile(self):
        profile_dir = os.path.join(os.getcwd(), "user_profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{self.username}.json")
        with open(profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=4, cls=NumpyEncoder)
        print(f"User profile saved to {profile_path}")
    
    def finish_comparisons(self):
        # Calculate ELO ratings based on comparisons
        self.calculate_film_ratings()
        
        QMessageBox.information(self, "Comparisons Complete",
                               f"Thank you! We've recorded {self.comparison_count} film comparisons.")
        self.close()
        
        # Here you would launch the final recommendation screen
    
    def calculate_film_ratings(self):
        # Initialize ratings if not present
        if "film_ratings" not in self.user_profile:
            self.user_profile["film_ratings"] = {}
        
        # Initialize all seen films with a base rating
        for film_id in self.seen_movies:
            if film_id not in self.user_profile["film_ratings"]:
                self.user_profile["film_ratings"][film_id] = 1500  # Base ELO rating
        
        # Process all comparisons to update ratings
        for comp in self.user_profile.get("film_comparisons", []):
            film1_id = comp["film1_id"]
            film2_id = comp["film2_id"]
            preference = comp["preference"]
            
            # Skip if we don't have ratings for both films
            if film1_id not in self.user_profile["film_ratings"] or film2_id not in self.user_profile["film_ratings"]:
                continue
            
            # Get current ratings
            rating1 = self.user_profile["film_ratings"][film1_id]
            rating2 = self.user_profile["film_ratings"][film2_id]
            
            # Calculate expected scores
            expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
            expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
            
            # Determine actual scores based on preference
            if preference == "left":
                actual1, actual2 = 1, 0
            elif preference == "right":
                actual1, actual2 = 0, 1
            else:  # equal
                actual1, actual2 = 0.5, 0.5
            
            # Update ratings (K-factor of 32)
            k_factor = 32
            self.user_profile["film_ratings"][film1_id] = rating1 + k_factor * (actual1 - expected1)
            self.user_profile["film_ratings"][film2_id] = rating2 + k_factor * (actual2 - expected2)
        
        # Save updated ratings
        self.save_user_profile()

class PopularMoviesScreen(QMainWindow):
    def __init__(self, username, user_profile, movie_data):
        super().__init__()
        self.username = username
        self.user_profile = user_profile
        self.movie_data = movie_data
        self.not_seen_count = 0  # Initialize the counter here

        
        # Get liked genres
        self.liked_genres = [genre for genre, rating in 
                            self.user_profile.get("genre_preferences", {}).items() 
                            if rating == 1]
        
        if not self.liked_genres:
            QMessageBox.information(self, "No Liked Genres", 
                                   "You didn't select any genres you like. Please try again.")
            self.close()
            return
            
        # Get popular movies for each liked genre
        self.genre_movies = {}
        for genre in self.liked_genres:
            # Filter movies where this genre is the primary genre (genre_1)
            genre_films = self.movie_data[self.movie_data['genre_1'] == genre]
            # Sort by popularity and take more movies (e.g., top 20 instead of 10)
            top_films = genre_films.sort_values('popularity', ascending=False).head(20)
            if not top_films.empty:
                self.genre_movies[genre] = top_films
        
        # Setup for movie display
        self.current_genre_index = 0
        self.current_movie_index = 0
        self.current_genre = self.liked_genres[0]
        
        # Initialize UI
        self.setWindowTitle("Popular Movies")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Genre header
        self.genre_header = QLabel(f"Genre: {self.current_genre}")
        self.genre_header.setAlignment(Qt.AlignCenter)
        self.genre_header.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(self.genre_header)
        
        # Movie display
        self.movie_display = QVBoxLayout()
        
        self.movie_title = QLabel("")
        self.movie_title.setAlignment(Qt.AlignCenter)
        self.movie_title.setFont(QFont("Arial", 18, QFont.Bold))
        self.movie_display.addWidget(self.movie_title)
        
        self.movie_details = QLabel("")
        self.movie_details.setAlignment(Qt.AlignCenter)
        self.movie_details.setWordWrap(True)
        self.movie_display.addWidget(self.movie_details)
        
        main_layout.addLayout(self.movie_display)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Haven't seen button (left)
        self.not_seen_button = QPushButton("Haven't Seen")
        self.not_seen_button.setFixedSize(150, 50)
        self.not_seen_button.clicked.connect(lambda: self.record_movie_status("not_seen"))
        buttons_layout.addWidget(self.not_seen_button)
        
        # Have seen button (right)
        self.seen_button = QPushButton("Have Seen")
        self.seen_button.setFixedSize(150, 50)
        self.seen_button.clicked.connect(lambda: self.record_movie_status("seen"))
        buttons_layout.addWidget(self.seen_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Progress indicator
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_label)
        
        # Set up keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.record_movie_status("not_seen"))
        
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.record_movie_status("seen"))
        
        # Initialize movie display
        self.show_current_movie()
    
    def show_current_movie(self):
        if self.current_genre not in self.genre_movies or not self.genre_movies[self.current_genre].shape[0]:
            # Skip to next genre if no movies for current genre
            self.move_to_next_genre()
            return
            
        if self.current_movie_index >= len(self.genre_movies[self.current_genre]):
            # Move to next genre when done with current genre's movies
            self.move_to_next_genre()
            return
            
        # Get current movie
        movie = self.genre_movies[self.current_genre].iloc[self.current_movie_index]
        
        # Update display
        self.movie_title.setText(movie['title'])
        
        # Create details text with release year, director, etc.
        details = f"Released: {movie['release_date'][:4]}\n"
        details += f"Director: {movie['director']}\n"
        details += f"Overview: {movie['overview'][:200]}..." if len(movie['overview']) > 200 else f"Overview: {movie['overview']}"
        
        self.movie_details.setText(details)
        
        # Update progress
        total_movies = len(self.genre_movies[self.current_genre])
        self.progress_label.setText(f"Movie {self.current_movie_index + 1} of {total_movies} in {self.current_genre}")
        
        # Update genre header
        self.genre_header.setText(f"Genre: {self.current_genre}")
    
    def move_to_next_genre(self):
        self.current_genre_index += 1
        self.not_seen_count = 0  # Reset the counter when moving to a new genre
        
        if self.current_genre_index < len(self.liked_genres):
            # Move to next genre
            self.current_genre = self.liked_genres[self.current_genre_index]
            self.current_movie_index = 0
            self.show_current_movie()
        else:
            # All genres processed
            self.finish_movie_selection()

    
    def record_movie_status(self, status):
        if self.current_genre not in self.genre_movies or self.current_movie_index >= len(self.genre_movies[self.current_genre]):
            return
            
        movie = self.genre_movies[self.current_genre].iloc[self.current_movie_index]
        movie_id = str(movie.name)  # Assuming the index is the movie ID
        
        # Initialize seen_movies if not present
        if "seen_movies" not in self.user_profile:
            self.user_profile["seen_movies"] = {}
            
        # Record if user has seen the movie
        if status == "seen":
            self.user_profile["seen_movies"][movie_id] = {
                "title": movie['title'],
                "genres": [self.current_genre],  # Just the primary genre for now
                "rating": None,  # Can be filled in later
                "timestamp": time.time()
            }
            # Note: We don't reset the not_seen_count here anymore
        else:
            # Increment the not_seen counter
            self.not_seen_count += 1
            
            # If they haven't seen 4 movies in this genre (cumulative), move to next genre
            if self.not_seen_count >= 4:
                # No judgmental message, just move to next genre
                self.not_seen_count = 0  # Reset counter
                self.move_to_next_genre()
                return
        
        # Move to next movie
        self.current_movie_index += 1
        self.show_current_movie()
        
        # Save profile after each selection
        self.save_user_profile()
    
    def save_user_profile(self):
        profile_dir = os.path.join(os.getcwd(), "user_profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{self.username}.json")
        with open(profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=4, cls=NumpyEncoder)
        print(f"User profile saved to {profile_path}")
    
    def finish_movie_selection(self):
        QMessageBox.information(self, "Movies Recorded",
                            f"Thank you {self.username}! We've recorded which movies you've seen.")
        
        # Load similarity matrix
        similarity_matrix = load_similarity_matrix()
        
        # Launch film comparison screen
        self.comparison_screen = FilmComparisonScreen(self.username, self.user_profile, 
                                                    similarity_matrix, self.movie_data)
        self.comparison_screen.show()
        self.close()

class GenrePreferenceScreen(QMainWindow):
    def __init__(self, username, user_profile):
        super().__init__()
        self.username = username
        self.user_profile = user_profile
        self.genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", 
                      "Documentary", "Drama", "Family", "Fantasy", "History", 
                      "Horror", "Music", "Mystery", "Romance", "Science Fiction", 
                      "Thriller", "TV Movie", "War", "Western"]
        self.current_genre_index = 0
        
        self.setWindowTitle("Genre Preferences")
        self.setGeometry(100, 100, 600, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Instructions
        instructions = QLabel("Please indicate your preference for each genre:")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 12))
        main_layout.addWidget(instructions)
        
        # Genre display
        self.genre_label = QLabel(self.genres[0])
        self.genre_label.setAlignment(Qt.AlignCenter)
        self.genre_label.setFont(QFont("Arial", 24, QFont.Bold))
        main_layout.addWidget(self.genre_label)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Dislike button (left)
        self.dislike_button = QPushButton("Dislike")
        self.dislike_button.setFixedSize(120, 50)
        self.dislike_button.clicked.connect(lambda: self.record_preference("dislike"))
        buttons_layout.addWidget(self.dislike_button)

        # Like button (right)
        self.like_button = QPushButton("Like")
        self.like_button.setFixedSize(120, 50)
        self.like_button.clicked.connect(lambda: self.record_preference("like"))
        buttons_layout.addWidget(self.like_button)

        main_layout.addLayout(buttons_layout)

        
        # Progress indicator
        self.progress_label = QLabel(f"1 of {len(self.genres)}")
        self.progress_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_label)
        
        # Set up keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.record_preference("dislike"))
        
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.record_preference("like"))
        
        # Initialize genre preferences in user profile if not present
        if "genre_preferences" not in self.user_profile:
            self.user_profile["genre_preferences"] = {}
    
    def record_preference(self, preference):
        # Check if we've already gone through all genres
        if self.current_genre_index >= len(self.genres):
            # We're done with all genres, just launch the main app
            self.launch_main_app()
            return
            
        current_genre = self.genres[self.current_genre_index]
        
        # Record preference
        if preference == "like":
            self.user_profile["genre_preferences"][current_genre] = 1
        elif preference == "dislike":
            self.user_profile["genre_preferences"][current_genre] = -1
        
        # Save user profile
        self.save_user_profile()
        
        # Move to next genre or finish
        self.current_genre_index += 1
        if self.current_genre_index < len(self.genres):
            self.genre_label.setText(self.genres[self.current_genre_index])
            self.progress_label.setText(f"{self.current_genre_index + 1} of {len(self.genres)}")
        else:
            # All genres processed, move to main application
            self.launch_main_app()

    
    def save_user_profile(self):
        profile_dir = os.path.join(os.getcwd(), "user_profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{self.username}.json")
        with open(profile_path, 'w') as f:
            json.dump(self.user_profile, f, indent=4, cls=NumpyEncoder)
        print(f"User profile saved to {profile_path}")
    
    def launch_main_app(self):
        # Load movie data
        movie_data_path = os.path.join(os.getcwd(), "tmdb_top_rated_movies.csv")
        movie_data = pd.read_csv(movie_data_path)
        
        # Launch popular movies screen
        self.popular_movies_screen = PopularMoviesScreen(self.username, self.user_profile, movie_data)
        self.popular_movies_screen.show()
        self.close()

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movie Recommender - Login")
        self.setGeometry(100, 100, 400, 200)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title label
        title_label = QLabel("Movie Recommender Login")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        main_layout.addWidget(title_label)
        
        # Username input
        form_layout = QFormLayout()
        self.username_input = QLineEdit()
        self.username_input.returnPressed.connect(self.login)
        form_layout.addRow("Username:", self.username_input)
        main_layout.addLayout(form_layout)
        
        # Login button
        login_button = QPushButton("Login")
        login_button.clicked.connect(self.login)
        main_layout.addWidget(login_button)
        
        # Status message
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
    def login(self):
        username = self.username_input.text().strip()
        if not username:
            self.status_label.setText("Please enter a username")
            return
            
        # Create user profile if it doesn't exist
        user_profile, is_new_user = self.load_or_create_profile(username)
        
        if is_new_user:
            # Launch genre preference screen for new users
            self.genre_screen = GenrePreferenceScreen(username, user_profile)
            self.genre_screen.show()
            self.close()
        else:
            # Check if user has completed onboarding
            has_genre_preferences = "genre_preferences" in user_profile and user_profile["genre_preferences"]
            has_seen_movies = "seen_movies" in user_profile and user_profile["seen_movies"]
            has_film_ratings = "film_ratings" in user_profile and user_profile["film_ratings"]
            
            # Load necessary data
            movie_data_path = os.path.join(os.getcwd(), "tmdb_top_rated_movies.csv")
            movie_data = pd.read_csv(movie_data_path)
            movie_data = preprocess_movie_data(movie_data)
            similarity_matrix = load_similarity_matrix()
            
            if has_genre_preferences and has_seen_movies and has_film_ratings:
                # User has completed onboarding, go directly to recommendations
                self.recommendation_screen = FilmRecommendationScreen(
                    username, user_profile, similarity_matrix, movie_data)
                self.recommendation_screen.show()
                self.close()
            elif has_genre_preferences and has_seen_movies:
                # User has seen movies but hasn't done comparisons
                self.comparison_screen = FilmComparisonScreen(
                    username, user_profile, similarity_matrix, movie_data)
                self.comparison_screen.show()
                self.close()
            elif has_genre_preferences:
                # User has genre preferences but hasn't seen movies
                self.popular_movies_screen = PopularMoviesScreen(
                    username, user_profile, movie_data)
                self.popular_movies_screen.show()
                self.close()
            else:
                # User profile exists but is incomplete, restart onboarding
                self.genre_screen = GenrePreferenceScreen(username, user_profile)
                self.genre_screen.show()
                self.close()


    def load_or_create_profile(self, username):
        profile_dir = os.path.join(os.getcwd(), "user_profiles")
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, f"{username}.json")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                return json.load(f)
        else:
            print("Creating a new profile...")
            return {}

def run_gui_app():
    # Load data and compute/load similarity matrix
    movie_data_path = os.path.join(os.getcwd(), "tmdb_top_rated_movies.csv")
    movie_data = pd.read_csv(movie_data_path)
    movie_data = preprocess_movie_data(movie_data)
    
    similarity_matrix = load_similarity_matrix()
    if similarity_matrix is None:
        # Compute similarity matrix using methods from holygoon39.py
        similarity_matrix = compute_similarity_matrix(movie_data)
        save_similarity_matrix(similarity_matrix)
    
    # Launch app
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui_app()


