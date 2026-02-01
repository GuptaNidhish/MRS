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

# Load TMDB API key from secrets or .env
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
    index = faiss.read_index("models/models_including_lang/faiss_lang.index")
    X_reduced = joblib.load("models/models_including_lang/movie_vectors_reduced_lang.pkl")
    movies = pd.read_csv(
        "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/data/processed/processed_language/reduce_processed_data_lang.csv"
    )
    return index, X_reduced, movies

index, X_reduced, movies = load_models()

# -----------------------------
# TMDB HELPERS
# -----------------------------
def fetch_ott_providers(movie_id, api_key, region="IN"):
    base_url = "https://api.themoviedb.org/3"
    provider_url = f"{base_url}/movie/{movie_id}/watch/providers"
    params = {"api_key": api_key}
    
    try:
        response = requests.get(provider_url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', {})
        if region not in results:
            return "No providers found"
        
        country_data = results[region]
        parts = []

        if 'flatrate' in country_data:
            streaming = ", ".join([p['provider_name'] for p in country_data['flatrate']])
            parts.append(f"Streaming: {streaming}")
        if 'rent' in country_data:
            rent = ", ".join([p['provider_name'] for p in country_data['rent']])
            parts.append(f"Rent: {rent}")
        if 'buy' in country_data:
            buy = ", ".join([p['provider_name'] for p in country_data['buy']])
            parts.append(f"Buy: {buy}")
        
        return " | ".join(parts) if parts else "No providers found"

    except requests.exceptions.RequestException as e:
        return f"API Error: {e}"

# -----------------------------
# SESSION CACHE (OTT by TMDB ID)
# -----------------------------
if "movie_ott_cache" not in st.session_state:
    st.session_state.movie_ott_cache = {}

def fetch_movie_ott(tmdb_id):
    return fetch_ott_providers(tmdb_id, TMDB_API_KEY)

# -----------------------------
# PARALLEL OTT FETCH + PROGRESS
# -----------------------------
def fetch_ott_parallel_with_progress(tmdb_ids):
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
            background-color: #E50914;
        }
        </style>
        """, unsafe_allow_html=True
    )
    ott_details = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_movie_ott, tmdb_id): tmdb_id for tmdb_id in tmdb_ids}
        completed = 0

        for future in as_completed(futures):
            tmdb_id = futures[future]
            try:
                result = future.result()   # ✅ NOW THIS WILL NOT FAIL
                ott_details[tmdb_id] = result
            except Exception as e:
                ott_details[tmdb_id] = f"Error: {e}"

            completed += 1
            progress_bar.progress(completed / len(tmdb_ids))
            status_text.caption(f"Checking OTT availability ({completed}/{len(tmdb_ids)})")

    progress_bar.empty()
    status_text.empty()
    return ott_details


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
# UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Content-based filtering using **SVD + FAISS**")
st.divider()

selected_movie = st.selectbox("Select a movie", movies["title"].values)
num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):
    # ----------------- Selected Movie -----------------
    selected_row = movies[movies["title"] == selected_movie].iloc[0]
    selected_poster = fetch_poster(selected_row["poster_path"])
    selected_rating = selected_row["imdb_rating"]
    selected_tmdb_id = selected_row["id"]  # Make sure this is TMDB ID
    selected_ott = fetch_movie_ott(selected_tmdb_id)

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.subheader("You Selected")
        if selected_poster:
            st.image(selected_poster, use_container_width=True)
        st.caption(selected_movie)
        with st.expander("View details"):
            st.markdown(f"**IMDb Rating:** {selected_rating}")
            st.markdown(f"**Available on:** {selected_ott}")

    # ----------------- Recommendations -----------------
    with right_col:
        st.subheader("Recommended Movies")
        recommendations = recommend(selected_movie, num_recommendations)
        tmdb_ids = recommendations["id"].tolist()
        ott_dict = fetch_ott_parallel_with_progress(tmdb_ids)
        for tmdb_id, ott in ott_dict.items():
            st.session_state.movie_ott_cache[tmdb_id] = ott
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
                        st.markdown(f"**IMDb Rating:** {movie['imdb_rating']}")
                        st.markdown(f"**Available on:** {ott_dict.get(movie['id'], 'Not available')}")
