import streamlit as st
import faiss
import joblib
import numpy as np
import pandas as pd
import requests

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide"
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# -----------------------------
# LOAD MODELS (cached)
# -----------------------------
@st.cache_resource
def load_models():
    index = faiss.read_index("/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/models/models_including_lang/faiss_lang.index")
    X_reduced = joblib.load("/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/models/models_including_lang/movie_vectors_reduced_lang.pkl")
    movies = pd.read_csv("/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem/data/processed/final_data.csv")

    return index, X_reduced, movies

index, X_reduced, movies = load_models()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fetch_poster(poster_path):
    if pd.isna(poster_path) or poster_path == "":
        return None
    return TMDB_IMAGE_BASE + poster_path

def recommend(movie_title, k=10):
    idx = movies[movies['title'] == movie_title].index[0]
    query_vector = X_reduced[idx].reshape(1, -1)
    distances, indices = index.search(query_vector, k + 1)
    rec_indices = indices[0][1:]
    return movies.iloc[rec_indices]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Content-based recommendations using **SVD + FAISS**")
st.divider()

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend 🚀"):
    # -----------------------------
    # Side-by-side layout
    # -----------------------------
    selected_idx = movies[movies['title'] == selected_movie].index[0]
    selected_poster_url = fetch_poster(movies.iloc[selected_idx]['poster_path'])

    left_col, right_col = st.columns([1, 3])  # left smaller, right bigger

    with left_col:
        st.subheader("You Selected")
        if selected_poster_url:
            st.image(selected_poster_url, use_container_width=True)
        st.caption(selected_movie)

    with right_col:
        st.subheader("Recommended Movies")
        recommendations = recommend(selected_movie, num_recommendations)
        cols = st.columns(5)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 5]:
                poster_url = fetch_poster(row['poster_path'])
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                st.caption(row['title'])
