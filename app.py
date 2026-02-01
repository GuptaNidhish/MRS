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

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

if "TMDB_API_KEY" in st.secrets:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
else:
    load_dotenv()
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    st.error("TMDB API key not found. Check secrets or .env file.")
    st.stop()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    index = faiss.read_index(
        "models/models_including_lang/faiss_lang.index"
    )
    X_reduced = joblib.load(
        "models/models_including_lang/movie_vectors_reduced_lang.pkl"
    )
    movies = pd.read_csv(
        "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/data/processed/processed_language/reduce_processed_data_lang.csv"
    )
    return index, X_reduced, movies

index, X_reduced, movies = load_models()

# -----------------------------
# TMDB HELPERS (OTT ONLY)
# -----------------------------
def safe_requests_get(url, params=None, timeout=6):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return {}

def fetch_ott(tmdb_id, country="IN"):
    if pd.isna(tmdb_id):
        return "Not available"

    url = f"{TMDB_BASE}/movie/{int(tmdb_id)}/watch/providers"
    params = {"api_key": TMDB_API_KEY}
    data = safe_requests_get(url, params)

    providers = (
        data.get("results", {})
        .get(country, {})
        .get("flatrate", [])
    )

    if not providers:
        return "Not available"

    return ", ".join(p["provider_name"] for p in providers)

# -----------------------------
# SESSION CACHE (OTT by TMDB ID)
# -----------------------------
if "movie_ott_cache" not in st.session_state:
    st.session_state.movie_ott_cache = {}

def fetch_movie_ott_cached(tmdb_id):
    cache = st.session_state.movie_ott_cache
    if tmdb_id in cache:
        return cache[tmdb_id]

    ott = fetch_ott(tmdb_id)
    cache[tmdb_id] = ott
    return ott

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
# PARALLEL OTT FETCH + PROGRESS
# -----------------------------
def fetch_ott_parallel_with_progress(tmdb_ids):
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

    ott_details = {}
    total = len(tmdb_ids)
    progress = st.progress(0)
    text = st.empty()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_movie_ott_cached, tmdb_id): tmdb_id
            for tmdb_id in tmdb_ids
        }

        completed = 0
        for future in as_completed(futures):
            tmdb_id = futures[future]
            try:
                ott_details[tmdb_id] = future.result()
            except:
                ott_details[tmdb_id] = "Not available"

            completed += 1
            progress.progress(completed / total)
            text.caption(f"Fetching OTT info {completed}/{total}")
            time.sleep(0.08)

    progress.empty()
    text.empty()
    return ott_details

# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Content-based filtering using **SVD + FAISS**")
st.divider()

selected_movie = st.selectbox(
    "Select a movie",
    movies["title"].values
)

num_recommendations = st.slider(
    "Number of recommendations",
    5, 20, 10
)

if st.button("Recommend"):
    selected_row = movies[movies["title"] == selected_movie].iloc[0]

    selected_poster = fetch_poster(selected_row["poster_path"])
    selected_rating = selected_row["imdb_rating"]
    selected_ott = fetch_movie_ott_cached(selected_row["id"])

    left_col, right_col = st.columns([1, 3])

    # SELECTED MOVIE
    with left_col:
        st.subheader("You Selected")
        if selected_poster:
            st.image(selected_poster, use_container_width=True)

        st.caption(selected_movie)

        with st.expander("View details"):
            st.markdown(f"**IMDb Rating:** {selected_rating}")
            st.markdown(f"**Available on:** {selected_ott}")

    # RECOMMENDATIONS
    with right_col:
        st.subheader("Recommended Movies")

        recommendations = recommend(
            selected_movie,
            num_recommendations
        )

        tmdb_ids = recommendations["id"].tolist()
        ott_dict = fetch_ott_parallel_with_progress(tmdb_ids)

        for i in range(0, len(recommendations), 5):
            row = recommendations.iloc[i:i+5]
            cols = st.columns(len(row))

            for col, (_, movie) in zip(cols, row.iterrows()):
                with col:
                    poster = fetch_poster(movie["poster_path"])
                    if poster:
                        st.image(poster, use_container_width=True)

                    st.caption(movie["title"])

                    with st.expander("View details"):
                        st.markdown(
                            f"**IMDb Rating:** {movie['imdb_rating']}"
                        )
                        st.markdown(
                            f"**Available on:** {ott_dict.get(movie['id'], 'Not available')}"
                        )
