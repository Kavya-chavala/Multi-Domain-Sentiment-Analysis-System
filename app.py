
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Multi-Domain Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Main App Background */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#172554,#0f172a,#1e293b);
color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
background: rgba(15,23,42,0.88);
backdrop-filter: blur(10px);
border-right:1px solid rgba(255,255,255,0.08);
}

/* Sidebar Text */
section[data-testid="stSidebar"] *{
color:white !important;
}

/* Cards */
.card{
background: rgba(30,41,59,0.65);
backdrop-filter: blur(12px);
padding:25px;
border-radius:18px;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0px 8px 30px rgba(0,0,0,0.25);
margin-bottom:20px;
}

/* Main Title */
.title{

font-size:52px;
font-weight:700;
text-align:center;
color:#60a5fa;
margin-bottom:10px;
}

/* Subtitle */
.subtitle{
text-align:center;
font-size:18px;
color:#cbd5e1;
margin-bottom:40px;
}

/* Section Titles */
.section-title{
font-size:30px;
font-weight:600;
margin-top:25px;
margin-bottom:20px;
color:#38bdf8;
}

/* Page Titles */
.main-title{
font-size:40px;
font-weight:700;
margin-bottom:20px;
color:#60a5fa;
}

/* General Text */
h1,h2,h3,h4,h5,h6,p,li,label,span{
color:#f8fafc !important;
}

/* Buttons */
.stButton>button{
background: linear-gradient(90deg,#22c55e,#16a34a);
color:white;
border:none;
border-radius:10px;
padding:12px 28px;
font-weight:600;
font-size:16px;
box-shadow:0px 4px 15px rgba(34,197,94,0.3);
transition:0.3s;
}

.stButton>button:hover{
transform:translateY(-2px);
box-shadow:0px 6px 20px rgba(34,197,94,0.45);
}

/* Text Area */
textarea{
background: rgba(15,23,42,0.8) !important;
color:white !important;
border:1px solid rgba(255,255,255,0.12) !important;
border-radius:12px !important;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"]{
background: rgba(15,23,42,0.9) !important;
border-radius:10px !important;
border:1px solid rgba(255,255,255,0.1) !important;
color:white !important;
}

/* Metrics */
[data-testid="metric-container"]{
background: rgba(30,41,59,0.55);
border:1px solid rgba(255,255,255,0.08);
padding:18px;
border-radius:16px;
}

/* Dataframes */
[data-testid="stDataFrame"]{
background: rgba(30,41,59,0.55);
border-radius:14px;
padding:10px;
}

/* File uploader */
[data-testid="stFileUploader"]{
background: rgba(30,41,59,0.45);
padding:20px;
border-radius:14px;
border:1px dashed rgba(255,255,255,0.15);
}
            
/* Selectbox selected value */
.stSelectbox div[data-baseweb="select"] > div{
color:white !important;
background:rgba(15,23,42,0.95) !important;
}

/* Dropdown menu options */
div[role="listbox"]{
background:#0f172a !important;
color:white !important;
}

/* Individual dropdown items */
div[role="option"]{
color:white !important;
background:#0f172a !important;
}

/* Hover effect */
div[role="option"]:hover{
background:#1e293b !important;
}
            
/* Dropdown popup container */
div[data-baseweb="popover"]{
background:#0f172a !important;
}

/* Dropdown options text */
li{
color:white !important;
}

/* Dropdown option background */
ul{
background:#0f172a !important;
}

/* Hovered option */
li:hover{
background:#1e293b !important;
}

            /* File uploader main box */
[data-testid="stFileUploader"] section{
background:#0f172a !important;
border:1px dashed rgba(255,255,255,0.15) !important;
color:white !important;
}

/* Upload text */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div{
color:white !important;
}

/* Browse files button */
[data-testid="stFileUploader"] button{
background:#1e293b !important;
color:white !important;
border:none !important;
}

/* Hover */
[data-testid="stFileUploader"] button:hover{
background:#334155 !important;
}



</style>
""", unsafe_allow_html=True)


# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("🔎 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home",
     "✍️ Single Prediction",
     "📂 Batch Prediction",
     "📊 Model Performance",
     "📘 About"]
)

st.sidebar.title("⚙️ Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Naive Bayes", "Logistic Regression", "Transformer (Advanced)"]
)

# Load selected model dynamically
if model_choice == "Naive Bayes":
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

elif model_choice == "Logistic Regression":
    model = joblib.load("model_logistic.pkl")
    vectorizer = joblib.load("vectorizer_logistic.pkl")

else:
    model = pipeline("sentiment-analysis")
    vectorizer = None



# -----------------------
# HOME PAGE
# -----------------------
if page == "🏠 Home":
    
    st.markdown('<div class="title">Multi-Domain Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI powered sentiment detection using multiple models</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title">🚀 Models Used</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🧠 Naive Bayes</h3>
        Fast classical machine learning sentiment model.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>📈 Logistic Regression</h3>
        Improved linear ML model for better accuracy.
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>🤖 Transformer</h3>
        Deep learning model using contextual embeddings.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Naive Bayes", "81.68%")
    col2.metric("Logistic Regression", "85%")
    col3.metric("Transformer", "~94%")

    st.markdown("---")

    st.markdown('<div class="section-title">📂 Dataset Information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <b>🎬 IMDB Reviews</b><br>
        Large dataset of movie reviews with sentiment labels.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <b>🐦 Twitter Dataset</b><br>
        Social media sentiment dataset used for real-world testing.
        </div>
        """, unsafe_allow_html=True)
elif page == "✍️ Single Prediction":

    st.markdown('<div class="main-title">✍️ Single Text Sentiment Analysis</div>', unsafe_allow_html=True)
    st.info(f"Currently using: **{model_choice}**")

    user_input = st.text_area("Enter your text here:", height=150)

    if st.button("Analyze Sentiment"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")

        else:

            st.markdown("---")
            st.markdown("## 🔍 Prediction Result")

            col1, col2 = st.columns(2)

            # 🤖 TRANSFORMER MODEL
            if model_choice == "Transformer (Advanced)":

                result = model(user_input)[0]
                prediction = result["label"]
                confidence = result["score"] * 100

                with col1:
                    if prediction == "POSITIVE":
                        st.success("### 😊 POSITIVE")
                    else:
                        st.error("### 😠 NEGATIVE")

                    st.metric("Confidence", f"{confidence:.2f}%")

                # Transformer does not give full probability distribution
                with col2:
                    st.write("### 📊 Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Sentiment": [prediction],
                        "Probability": [confidence / 100]
                    })
                    st.bar_chart(prob_df.set_index("Sentiment"))

            # 🧮 CLASSICAL MODELS
            else:

                vectorized_input = vectorizer.transform([user_input])

                prediction = model.predict(vectorized_input)[0]
                probabilities = model.predict_proba(vectorized_input)[0]
                classes = model.classes_

                confidence = max(probabilities) * 100

                with col1:
                    if prediction == "positive":
                        st.success("### 😊 POSITIVE")
                    else:
                        st.error("### 😠 NEGATIVE")

                    st.metric("Confidence", f"{confidence:.2f}%")

                with col2:
                    st.write("### 📊 Probability Distribution")

                    prob_df = pd.DataFrame({
                        "Sentiment": classes,
                        "Probability": probabilities
                    })

                    st.bar_chart(prob_df.set_index("Sentiment"))

# -----------------------
# BATCH PREDICTION
# -----------------------

elif page == "📂 Batch Prediction":

    st.markdown('<div class="main-title">📂 Batch Sentiment Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    Upload a CSV file with a column named **review**.
    The model will predict sentiment for each row.
    """)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        if "review" not in data.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            st.success("File uploaded successfully!")

            # 🤖 TRANSFORMER MODEL
            if model_choice == "Transformer (Advanced)":

                predictions = []
                confidences = []

                for text in data["review"].astype(str):
                    result = model(text)[0]
                    predictions.append(result["label"])
                    confidences.append(round(result["score"] * 100, 2))

                data["Predicted_Sentiment"] = predictions
                data["Confidence (%)"] = confidences

            # 🧮 CLASSICAL MODELS
            else:

                vectorized = vectorizer.transform(data["review"].astype(str))

                predictions = model.predict(vectorized)
                probabilities = model.predict_proba(vectorized)

                data["Predicted_Sentiment"] = predictions
                data["Confidence (%)"] = (probabilities.max(axis=1) * 100).round(2)

            

            # Show results
            st.markdown("### 🔍 Prediction Results")
            st.dataframe(data.head())

            # Sentiment distribution
            st.markdown("### 📊 Sentiment Distribution")
            sentiment_counts = data["Predicted_Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            # Download
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )
