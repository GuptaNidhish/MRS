import streamlit as st
import faiss
import joblib
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from dotenv import load_dotenv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

load_dotenv(dotenv_path=Path(".env"))

TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
print(TMDB_API_KEY)
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    st.error("TMDB API key not found. Check your .env file.")
    st.stop()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    index = faiss.read_index(
        "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/models/models_including_lang/faiss_lang.index"
    )
    X_reduced = joblib.load(
        "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/models/models_including_lang/movie_vectors_reduced_lang.pkl"
    )
    movies = pd.read_csv(
        "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/data/processed/final_data.csv"
    )
    return index, X_reduced, movies

index, X_reduced, movies = load_models()

# -----------------------------
# TMDB HELPERS
# -----------------------------
def safe_requests_get(url, params=None, timeout=6):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return {}

def imdb_to_tmdb(imdb_id):
    if pd.isna(imdb_id):
        return None
    url = f"{TMDB_BASE}/find/{imdb_id}"
    params = {"api_key": TMDB_API_KEY, "external_source": "imdb_id"}
    data = safe_requests_get(url, params)
    results = data.get("movie_results", [])
    return results[0]["id"] if results else None

def fetch_rating(tmdb_id):
    if not tmdb_id:
        return "Unknown"
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY}
    data = safe_requests_get(url, params)
    vote = data.get("vote_average", 0)
    return round(vote, 1) if vote else "Unknown"

def fetch_ott(tmdb_id, country="IN"):
    if not tmdb_id:
        return "Not available"
    url = f"{TMDB_BASE}/movie/{tmdb_id}/watch/providers"
    params = {"api_key": TMDB_API_KEY}
    data = safe_requests_get(url, params)
    providers = data.get("results", {}).get(country, {}).get("flatrate", [])
    if not providers:
        return "Unknown"
    return ", ".join(p.get("provider_name", "Unknown") for p in providers)

# -----------------------------
# SESSION CACHE
# -----------------------------
if "movie_details_cache" not in st.session_state:
    st.session_state.movie_details_cache = {}

def fetch_movie_details_cached(imdb_id):
    cache = st.session_state.movie_details_cache
    if imdb_id in cache:
        return cache[imdb_id]

    tmdb_id = imdb_to_tmdb(imdb_id)
    if not tmdb_id:
        details = ("Unknown", "Unknown")
    else:
        rating = fetch_rating(tmdb_id)
        ott = fetch_ott(tmdb_id)
        details = (rating, ott)

    cache[imdb_id] = details
    return details

# -----------------------------
# RECOMMENDER
# -----------------------------
def fetch_poster(path):
    if pd.isna(path) or path == "":
        return None
    return TMDB_IMAGE_BASE + path

def recommend(movie_title, k=10):
    idx = movies[movies["title"] == movie_title].index[0]
    query = X_reduced[idx].reshape(1, -1)
    k = min(k, len(movies) - 1)
    _, indices = index.search(query, k + 1)
    return movies.iloc[indices[0][1:]]

# -----------------------------
# PARALLEL FETCH + PROGRESS
# -----------------------------
def fetch_details_parallel_with_progress(imdb_ids):
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
            background-color: red;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    details = {}
    total = len(imdb_ids)
    progress = st.progress(0)
    text = st.empty()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_movie_details_cached, imdb_id): imdb_id
            for imdb_id in imdb_ids
        }

        completed = 0
        for future in as_completed(futures):
            imdb_id = futures[future]
            try:
                details[imdb_id] = future.result()
            except:
                details[imdb_id] = ("Unknown", "Unknown")

            completed += 1
            progress.progress(completed / total)
            text.caption(f"Fetching movie details {completed}/{total}")
            time.sleep(0.08)  # 👈 makes progress bar visible

    progress.empty()
    text.empty()
    return details

# -----------------------------
# UI
# -----------------------------
st.title("Movie Recommendation System")
st.markdown("Content-based filtering using **SVD + FAISS**")
st.divider()

selected_movie = st.selectbox("Select a movie", movies["title"].values)
num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend "):
    selected_idx = movies[movies["title"] == selected_movie].index[0]
    selected_movie_row = movies.iloc[selected_idx]

    selected_poster = fetch_poster(selected_movie_row["poster_path"])
    selected_rating, selected_ott = fetch_movie_details_cached(
        selected_movie_row["imdb_id"]
    )

    left_col, right_col = st.columns([1, 3])

    # SELECTED MOVIE
    with left_col:
        st.subheader("You Selected")
        if selected_poster:
            st.image(selected_poster, use_container_width=True)
        st.caption(selected_movie)

        with st.expander("View details"):
            st.markdown(f"**TMDb Rating:** {selected_rating}")
            st.markdown(f"**Available on:** {selected_ott}")

    # RECOMMENDATIONS
    with right_col:
        st.subheader("Recommended Movies ")
        recommendations = recommend(selected_movie, num_recommendations)

        imdb_ids = recommendations["imdb_id"].tolist()
        details_dict = fetch_details_parallel_with_progress(imdb_ids)

        for i in range(0, len(recommendations), 5):
            row = recommendations.iloc[i:i+5]
            cols = st.columns(len(row))
            for col, (_, movie) in zip(cols, row.iterrows()):
                with col:
                    poster = fetch_poster(movie["poster_path"])
                    if poster:
                        st.image(poster, use_container_width=True)
                    st.caption(movie["title"])

                    rating, ott = details_dict.get(
                        movie["imdb_id"], ("Unknown", "Not available")
                    )
                    with st.expander("View details"):
                        st.markdown(f"**TMDb Rating:** {rating}")
                        st.markdown(f"**Available on:** {ott}")
