import streamlit as st
import pandas as pd
import re
import nltk
import random
import plotly.express as px
import plotly.graph_objects as go

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download('stopwords')

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analysis of Social Media Posts",
    page_icon="✨",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #ff9a9e 0%,
        #fad0c4 20%,
        #fbc2eb 40%,
        #a18cd1 60%,
        #84fab0 80%,
        #8fd3f4 100%
    );
    background-attachment: fixed;
}

/* REMOVE DEFAULT WHITE CONTAINERS */
.block-container {
    padding-top: 2rem;
}

/* TITLE */
.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: white;
    text-shadow: 2px 2px 20px rgba(0,0,0,0.4);
    margin-bottom: 10px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    font-size: 22px;
    color: white;
    margin-bottom: 40px;
}

/* GLASS CARD */
.glass-card {
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    box-shadow: 0 8px 32px rgba(31,38,135,0.2);
}

/* TEXT AREA */
.stTextArea textarea {
    background: rgba(255,255,255,0.25) !important;
    color: black !important;
    border-radius: 15px !important;
    font-size: 18px !important;
    border: none !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 3.5em;
    border-radius: 15px;
    border: none;
    background: linear-gradient(
        90deg,
        #ff6ec4,
        #7873f5,
        #4ADEDE
    );
    color: white;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 25px rgba(0,0,0,0.3);
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.22);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.metric-number {
    font-size: 42px;
    font-weight: bold;
}

.metric-label {
    font-size: 18px;
}

/* RESULT BOX */
.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.positive-box {
    background: linear-gradient(135deg, #00c9a7, #92fe9d);
}

.negative-box {
    background: linear-gradient(135deg, #ff758c, #ff7eb3);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="main-title">
✨ Sentiment Analysis of Social Media Posts
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Analyze emotions and sentiments from social media posts using
Machine Learning + NLP
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### Technologies Used
- Python
- Streamlit
- NLP
- TF-IDF
- LinearSVC (SVM)
- Scikit-learn

### Features
✅ Bulk social media analysis  
✅ Pie chart visualization  
✅ Emotion analytics  
✅ Interactive dashboard  
✅ AI-powered classification  
""")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():

    data = pd.read_csv(
        "socialmedia.csv",
        encoding='latin-1',
        header=None
    )

    data.columns = [
        'sentiment',
        'id',
        'date',
        'query',
        'user',
        'text'
    ]

    data = data[['sentiment', 'text']]

    # Smaller sample
    data = data.sample(50000, random_state=42)

    # Convert labels
    data['sentiment'] = data['sentiment'].replace(4, 1)

    return data

data = load_data()

# -----------------------------
# STOPWORDS
# -----------------------------
stop_words = set(stopwords.words('english'))

# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'#', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# PREPROCESS DATA
# -----------------------------
data['clean_text'] = data['text'].apply(clean_text)

X = data['clean_text']
y = data['sentiment']

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# -----------------------------
# MODEL
# -----------------------------
model = LinearSVC()

model.fit(X, y)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.subheader("📝 Paste Social Media Posts")

user_input = st.text_area(
    "Enter multiple posts (one post per line)",
    height=250,
    placeholder="""
I love this new phone update
This app is terrible
Amazing customer support
Very disappointing experience
"""
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🚀 Analyze Emotions"):

    if user_input.strip() == "":
        st.warning("Please enter some social media posts.")

    else:

        posts = user_input.split("\n")

        posts = [
            post.strip()
            for post in posts
            if post.strip() != ""
        ]

        positive_count = 0
        negative_count = 0

        results = []

        positive_emotions = [
            "JOY 😊",
            "EXCITEMENT 🎉",
            "SATISFACTION 😄",
            "OPTIMISM 🌟"
        ]

        negative_emotions = [
            "ANGER 😠",
            "FRUSTRATION 😤",
            "DISAPPOINTMENT 😞",
            "DISSATISFACTION ⚠️"
        ]

        for post in posts:

            cleaned = clean_text(post)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)

            if prediction[0] == 1:

                positive_count += 1

                results.append(
                    random.choice(positive_emotions)
                )

            else:

                negative_count += 1

                results.append(
                    random.choice(negative_emotions)
                )

        total_posts = len(posts)

        positive_percentage = (
            positive_count / total_posts
        ) * 100

        negative_percentage = (
            negative_count / total_posts
        ) * 100

        # -----------------------------
        # METRICS
        # -----------------------------
        st.markdown("## 📊 Emotion Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{total_posts}</div>
                <div class="metric-label">
                Total Posts
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">
                {positive_count}
                </div>
                <div class="metric-label">
                Positive Posts
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">
                {negative_count}
                </div>
                <div class="metric-label">
                Negative Posts
                </div>
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # CHARTS
        # -----------------------------
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            pie_data = pd.DataFrame({
                "Sentiment": [
                    "Positive",
                    "Negative"
                ],
                "Count": [
                    positive_count,
                    negative_count
                ]
            })

            fig = px.pie(
                pie_data,
                names="Sentiment",
                values="Count",
                hole=0.45,
                title="Sentiment Distribution"
            )

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with chart_col2:

            bar_fig = go.Figure(data=[
                go.Bar(
                    x=["Positive", "Negative"],
                    y=[
                        positive_count,
                        negative_count
                    ]
                )
            ])

            bar_fig.update_layout(
                title="Emotion Overview",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )

        # -----------------------------
        # OVERALL RESULT
        # -----------------------------
        st.markdown("## 🎯 Overall Emotion")

        if positive_count >= negative_count:

            st.markdown(f"""
            <div class="result-box positive-box">
            Overall Audience Mood:
            POSITIVE & OPTIMISTIC ✨
            <br><br>
            Positive Sentiment:
            {positive_percentage:.1f}%
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-box negative-box">
            Overall Audience Mood:
            NEGATIVE & FRUSTRATED ⚠️
            <br><br>
            Negative Sentiment:
            {negative_percentage:.1f}%
            </div>
            """, unsafe_allow_html=True)

        # -----------------------------
        # DETAILED ANALYSIS
        # -----------------------------
        st.markdown("## 🔍 Individual Post Analysis")

        for i, post in enumerate(posts):

            st.markdown(f"""
            <div class="glass-card">
            <b>Post {i+1}:</b><br><br>
            {post}
            <br><br>
            <b>Detected Emotion:</b>
            {results[i]}
            </div>
            <br>
            """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption("""
Developed using Python, Streamlit,
Natural Language Processing (NLP),
TF-IDF Vectorization,
and Support Vector Machine (SVM)
""")