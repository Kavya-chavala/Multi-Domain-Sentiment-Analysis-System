
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
    layout="wide"
)


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
st.sidebar.markdown("---")
st.sidebar.title("⚙️ Model Selection")

model_choice = st.sidebar.radio(
    "Choose Model:",
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

    st.title("💬 Multi-Domain Sentiment Analysis System")

    st.markdown("""
    ### 🚀 Overview
    
    This application performs sentiment analysis using **three different models**:
    
    - 🧮 **Naive Bayes (Classical ML)**
    - 📈 **Logistic Regression (Improved Classical ML)**
    - 🤖 **Transformer (DistilBERT – Deep Learning)**
    
    The system is trained on:
    
    - 🎬 IMDB Movie Reviews  
    - 🐦 Twitter Sentiment Dataset  
    
    It supports:
    - Single text prediction  
    - Batch CSV prediction  
    - Model comparison  
    - Confidence visualization  
    """)

    st.markdown("---")

    # Model Performance Section
    st.subheader("📊 Model Performance Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Naive Bayes Accuracy", "81.68%")

    with col2:
        st.metric("Logistic Regression Accuracy", "84–86%")

    with col3:
        st.metric("Transformer (SST-2 Benchmark)", "~94%")

    st.markdown("---")

    # Dataset Info
    st.subheader("📂 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Samples", "60,000")

    with col2:
        st.metric("Domains", "Movies + Social Media")

    st.markdown("---")

    st.info("Use the sidebar to start analyzing sentiment using different models.")

elif page == "✍️ Single Prediction":

    st.title("✍️ Single Text Sentiment Analysis")
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

    st.title("📂 Batch Sentiment Analysis (CSV Upload)")

    st.markdown("""
    Upload a CSV file with a column named **review**.
    The model will predict sentiment for each row.
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

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
