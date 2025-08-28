import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
import base64
import logging
import time
from flask import Flask, render_template, request
from rapidfuzz import fuzz

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Spotify credentials
CLIENT_ID = "dc7fc0e0f88f4ebe925bb59e3bf77fd2"
CLIENT_SECRET = "17864a581a5c4a11821fc581685f8664"

# Load dataset
df = pd.read_csv("C:\\Users\\Shikhar\\Downloads\\data.csv")
df['artists'] = df['artists'].apply(eval)

# Normalize numerical columns for similarity computation
numerical_cols = ['valence', 'year', 'acousticness', 'danceability', 'energy',
                  'explicit', 'instrumentalness', 'liveness', 'loudness',
                  'mode', 'popularity', 'speechiness', 'tempo']
scaler = StandardScaler()
numerical_data_std = scaler.fit_transform(df[numerical_cols])

# Global token cache variables
cached_token = None
token_expires_at = 0

def get_access_token():
    """Obtain and cache an access token until it expires."""
    global cached_token, token_expires_at
    current_time = time.time()
    if cached_token and current_time < token_expires_at:
        return cached_token

    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()
    headers = {"Authorization": f"Basic {b64_auth_str}"}
    data = {"grant_type": "client_credentials"}
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    if response.status_code == 200:
        token_data = response.json()
        cached_token = token_data['access_token']
        token_expires_at = current_time + token_data['expires_in']  # usually 3600 seconds
        logging.info("Access token retrieved and cached.")
        return cached_token
    else:
        logging.error(f"Failed to get access token: {response.status_code}")
        return None

def get_track_details(track_ids, access_token):
    """Fetch track details from Spotify for given track IDs."""
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"ids": ",".join(track_ids)}
    response = requests.get("https://api.spotify.com/v1/tracks", headers=headers, params=params)
    if response.status_code == 200:
        return response.json()['tracks']
    else:
        logging.error(f"Error fetching track details: {response.status_code}")
        return [None] * len(track_ids)

def jaccard_similarity(list1, list2):
    """Compute the Jaccard similarity between two lists."""
    set1, set2 = set(list1), set(list2)
    union = set1 | set2
    if not union:
        return 0
    return len(set1 & set2) / len(union)

def fuzzy_match_song(input_value):
    """Attempt a fuzzy match for the song name using rapidfuzz."""
    scores = df['name'].apply(lambda name: fuzz.ratio(input_value.lower(), name.lower()))
    max_score = scores.max()
    if max_score > 80:  # threshold for a match
        return df.loc[scores.idxmax()]
    return None

def recommend_songs(input_value, data, numerical_data, alpha=0.9, n=5):
    """
    Recommend songs based on the input.
    
    Steps:
    1. Match input against artist and song name.
    2. If no exact match, try fuzzy matching.
    3. Compute cosine similarity on numerical features.
    4. Compute artist similarity (Jaccard).
    5. Add random noise to the similarity scores.
    6. Increase recommendation pool, then shuffle the top results.
    7. Return top n recommendations.
    """
    input_value = input_value.lower()
    artist_matches = data[data['artists'].apply(lambda x: input_value in [a.lower() for a in x])]
    song_matches = data[data['name'].str.lower().str.contains(input_value, na=False)]

    if artist_matches.empty and song_matches.empty:
        fuzzy_match = fuzzy_match_song(input_value)
        if fuzzy_match is not None:
            song_matches = data[data['name'] == fuzzy_match['name']]
        else:
            return None

    if not artist_matches.empty:
        artist_indices = artist_matches.index
        comparison_artists = [input_value]
    else:
        song_row = song_matches.iloc[0]
        artist_name = song_row['artists'][0].lower()
        artist_matches = data[data['artists'].apply(lambda x: artist_name in [a.lower() for a in x])]
        artist_indices = song_matches.index
        comparison_artists = [artist_name]

    # Compute cosine similarity between the selected songs and all songs
    sim_scores = cosine_similarity(numerical_data[artist_indices], numerical_data)
    avg_sim_scores = sim_scores.mean(axis=0)

    # Compute artist similarity using Jaccard similarity
    artist_sims = data['artists'].apply(lambda x: jaccard_similarity(comparison_artists, x))
    overall_sims = alpha * avg_sim_scores + (1 - alpha) * artist_sims
    overall_sims[artist_indices] = -1  # Exclude the input songs

    # Approach 2: Add a small random noise factor
    noise = np.random.uniform(-0.01, 0.01, overall_sims.shape)
    overall_sims_noisy = overall_sims + noise

    # Approach 3: Increase recommendation pool size, e.g., top 20
    pool_size = 20
    similar_indices = overall_sims_noisy.argsort()[::-1]
    top_indices = similar_indices[:pool_size]

    # Approach 1: Shuffle the top indices to add randomness
    top_indices = list(top_indices)
    np.random.shuffle(top_indices)

    recommended_songs = data.iloc[top_indices]
    # Filter out songs from the same artist for diversity
    recommended_songs = recommended_songs[
        ~recommended_songs['artists'].apply(lambda x: any(artist.lower() in [a.lower() for a in x] for artist in comparison_artists))
    ]
    return recommended_songs[['name', 'artists', 'year', 'id']].head(n).to_dict(orient='records')

@app.route("/", methods=["GET", "POST"])
def home():
    try:
        if request.method == "POST":
            user_input = request.form.get("search_query", "").strip()
            if user_input:
                logging.info(f"User search input: {user_input}")
                recommendations = recommend_songs(user_input, df, numerical_data_std)
                if recommendations:
                    access_token = get_access_token()
                    if not access_token:
                        logging.error("No access token available.")
                        return render_template("index.html", query=user_input, recommendations=[])
                    track_ids = [song['id'] for song in recommendations]
                    track_details = get_track_details(track_ids, access_token)
                    for song, details in zip(recommendations, track_details):
                        if details and details.get('album') and details['album'].get('images'):
                            song['album_cover'] = details['album']['images'][0]['url']
                            song['track_url'] = f"https://open.spotify.com/track/{song['id']}"
                        else:
                            song['album_cover'] = None
                            song['track_url'] = None
                else:
                    recommendations = []
                return render_template("index.html", query=user_input, recommendations=recommendations)
        return render_template("index.html", query=None, recommendations=[])
    except Exception as e:
        logging.exception("Unhandled exception in home route")
        return render_template("index.html", query=None, recommendations=[])

if __name__ == "__main__":
    app.run(debug=True)
