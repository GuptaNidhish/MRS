import streamlit as st
import pandas as pd
import faiss
import joblib
import numpy as np

# ---------------- CONFIG ----------------
TMDB_BASE_URL = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

# ---------------- PATHS ----------------
BASE_DIR = "/Users/nidhishgupta/Desktop/Movie_recommendation_sysytem"

FAISS_PATH = f"{BASE_DIR}/models/faiss.index"
VECTORS_PATH = f"{BASE_DIR}/models/movie_vectors_reduced.pkl"
DATA_PATH = f"{BASE_DIR}/data/processed/final_data.csv"

# ---------------- LOADERS ----------------
@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_PATH)

@st.cache_resource
def load_vectors():
    vectors = joblib.load(VECTORS_PATH)
    return vectors.astype("float32")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ---------------- LOAD ----------------
index = load_faiss_index()
vectors = load_vectors()
data = load_data()

# ---------------- SANITY CHECK (SAFE) ----------------
assert vectors.shape[0] == data.shape[0], "Vectors & data row mismatch"
assert vectors.shape[1] == index.d, "Vector dim & FAISS dim mismatch"

# ---------------- RECOMMENDER ----------------
def recommend(movie_title, k=10):
    idx_list = data.index[data["title"] == movie_title]

    if len(idx_list) == 0:
        return pd.DataFrame()

    idx = idx_list[0]

    query_vector = vectors[idx].reshape(1, -1)

    distances, indices = index.search(query_vector, k + 1)

    rec_indices = indices[0][1:]  # remove the movie itself
    return data.iloc[rec_indices]

# ---------------- UI ----------------

# Split layout: select box on left, poster on right
col1, col2 = st.columns([2, 1])

with col1:
    movie = st.selectbox(
        "Select a movie",
        data["title"].values
    )

with col2:
    # Show poster of selected movie
    selected_idx_list = data.index[data["title"] == movie]
    if len(selected_idx_list) > 0:
        poster_path = data.iloc[selected_idx_list[0]]["poster_path"]
        if pd.notna(poster_path):
            st.image(TMDB_BASE_URL + poster_path, use_column_width=True)

# Recommend button
if st.button("Recommend"):
    recs = recommend(movie, k=10)

    if recs.empty:
        st.warning("No recommendations found.")
    else:
        # Display recommendations in 10 columns
        cols = st.columns(10)
        for i, col in enumerate(cols):
            with col:
                st.text(recs.iloc[i]["title"])
                poster = recs.iloc[i]["poster_path"]
                if pd.notna(poster):
                    st.image(TMDB_BASE_URL + poster)
