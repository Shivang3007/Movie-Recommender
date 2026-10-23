import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MoodFlix | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    movies = joblib.load("movies.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    tfidf_vectors = joblib.load("tfidf_vectors.pkl")
    mood_genres = joblib.load("mood_genres.pkl")
    mood_tags = joblib.load("mood_tags.pkl")

    return movies, tfidf, tfidf_vectors, mood_genres, mood_tags


movies, tfidf, tfidf_vectors, mood_genres, mood_tags = load_data()


# ---------------- MOOD RECOMMENDATION ----------------
def recommend_mood(mood, number=5):
    selected_genres = mood_genres[mood]

    candidates = movies[
        movies["genres"].apply(
            lambda genres: any(
                genre in genres for genre in selected_genres
            )
        )
    ].copy()

    if candidates.empty:
        return candidates

    mood_vector = tfidf.transform([mood_tags[mood]])
    candidate_vectors = tfidf_vectors[candidates.index]

    scores = cosine_similarity(
        mood_vector,
        candidate_vectors
    ).flatten()

    candidates["mood_score"] = scores

    return candidates.sort_values(
        "mood_score",
        ascending=False
    ).head(number)


# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10101c, #19152d, #10101c);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #7c3aed;
    background-color: #5b21b6;
    color: white;
    padding: 12px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #7c3aed;
    border-color: #a78bfa;
}

[data-testid="stSelectbox"] {
    background-color: #211c35;
    border-radius: 10px;
}

.movie-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 22px;
    font-weight: 700;
    color: #c4b5fd;
}

.movie-info {
    color: #d1d5db;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- WEBSITE HEADER ----------------
st.title("🎬 MoodFlix")
st.markdown(
    "### Your mood. Your movie. Your moment."
)
st.write(
    "Tell us how you're feeling, and discover movies that match your vibe."
)

st.divider()


# ---------------- MOOD SELECTION ----------------
mood_options = {
    "😊 Happy": "Happy 😊",
    "😢 Sad": "Sad 😢",
    "😌 Stressed": "Stressed 😌",
    "🤩 Adventurous": "Adventurous 🤩",
    "😱 Thrilled": "Thrilled 😱",
    "❤️ Romantic": "Romantic ❤️",
    "👻 Scared": "Scared 👻",
    "🧠 Curious": "Curious 🧠"
}

col1, col2 = st.columns(2)

with col1:
    selected_label = st.selectbox(
        "How are you feeling today?",
        list(mood_options.keys())
    )

with col2:
    number = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

selected_mood = mood_options[selected_label]

if st.button("✨ Find My Movies"):
    results = recommend_mood(selected_mood, number)

    st.session_state["movie_results"] = results
    st.session_state["selected_mood"] = selected_mood


# ---------------- DISPLAY RESULTS ----------------
if "movie_results" in st.session_state:
    results = st.session_state["movie_results"]
    mood = st.session_state["selected_mood"]

    st.divider()
    st.subheader(f"🍿 Movies for your {mood} mood")

    if results.empty:
        st.warning("No matching movies found.")
    else:
        for _, movie in results.iterrows():
            title = movie["title"]
            genres = ", ".join(movie["genres"])
            score = float(movie["mood_score"])

            overview = movie["overview"]
            if isinstance(overview, list):
                overview = " ".join(overview)

            if not isinstance(overview, str) or not overview.strip():
                overview = "No description available."

            with st.container(border=True):
                st.markdown(
                    f'<div class="movie-title">{title}</div>',
                    unsafe_allow_html=True
                )

                st.caption(f"🎭 Genres: {genres}")
                st.progress(max(0.0, min(1.0, score)))
                st.caption(f"Mood similarity score: {score:.3f}")

                with st.expander("Read movie description"):
                    st.write(overview)

st.divider()
st.caption(
    "MoodFlix | Built with Python, Pandas, Scikit-learn and Streamlit"
)
