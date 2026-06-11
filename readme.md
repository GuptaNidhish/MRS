# 🎬 MRS - Movie Recommendation System

A scalable **Content-Based Movie Recommendation System** that recommends similar movies using movie metadata and semantic similarity techniques. The project leverages **TF-IDF**, **Truncated SVD**, and **FAISS** to efficiently process and search through a dataset containing over **215,000 movies**.

## 🚀 Features

* Processed and analyzed **215,000+ movies** with optimized memory usage.
* Reduced **23 raw dataset columns to 9 key attributes** for efficient feature engineering.
* Built semantic movie representations using **TF-IDF Vectorization**.
* Applied **Truncated SVD** for dimensionality reduction and faster similarity computations.
* Integrated **FAISS (Facebook AI Similarity Search)** for high-performance nearest-neighbor retrieval.
* Interactive **Streamlit web application** for real-time movie recommendations.
* Persisted trained artifacts including:

  * TF-IDF Vectorizer
  * Truncated SVD Model
  * FAISS Index

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **FAISS**
* **Streamlit**
* **Pickle / Joblib**

---

## 📊 Dataset

The system is designed to work with a large-scale movie dataset containing:

* Movie titles
* Genres
* Overview/Description
* Keywords
* Cast information
* Crew information
* Additional metadata

After preprocessing, the dataset was reduced from **23 columns to 9 meaningful features** to improve performance while preserving recommendation quality.

---

## ⚙️ How It Works

### 1. Data Preprocessing

* Clean missing values and duplicate records.
* Select the most informative movie attributes.
* Combine relevant textual features into a unified representation.

### 2. Feature Engineering

* Transform movie metadata into numerical vectors using **TF-IDF**.
* Capture semantic relationships between movies based on content.

### 3. Dimensionality Reduction

* Apply **Truncated SVD** to reduce vector dimensionality.
* Improve memory efficiency and search speed.

### 4. Similarity Search

* Build a **FAISS Index** on the reduced vectors.
* Retrieve nearest-neighbor movies in milliseconds.

### 5. Recommendation Generation

* User selects a movie.
* The system finds semantically similar movies.
* Top recommendations are displayed through the Streamlit interface.

---

## 📂 Project Structure

```text
MRS/
│
├── app.py                 # Streamlit application
├── data/
│   └── movies.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svd_model.pkl
│   └── faiss_index.bin
│
├── notebooks/
│   └── recommendation_pipeline.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── indexing.py
│   └── recommendation.py
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### Clone the Repository

```bash
git clone https://github.com/GuptaNidhish/MRS.git
cd MRS
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```text
http://localhost:8501
```

---

## 📈 Performance Highlights

* Successfully processed **215,000+ movie records**.
* Optimized memory consumption through feature reduction and dimensionality reduction.
* Enabled fast recommendation retrieval using **FAISS Approximate Nearest Neighbor Search**.
* Designed for scalability and real-time recommendation generation.

---

## 🎯 Example Workflow

1. Enter or select a movie title.
2. The movie metadata is transformed using the trained TF-IDF model.
3. The vector is projected into the latent semantic space using Truncated SVD.
4. FAISS searches for the nearest neighbors.
5. Top similar movies are returned instantly.

---

## 🔮 Future Improvements

* Hybrid recommendation system (Content + Collaborative Filtering)
* Movie poster and trailer integration
* User ratings and personalized recommendations
* API deployment using FastAPI
* Docker containerization
* Cloud deployment on AWS/GCP/Azure

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome. Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Nidhish Gupta**

* GitHub: https://github.com/GuptaNidhish
* LinkedIn: Add your LinkedIn profile here

⭐ If you found this project useful, consider giving it a star!
