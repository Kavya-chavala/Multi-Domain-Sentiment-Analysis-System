**Multi-Domain Sentiment Analysis System**

An AI-powered web application for sentiment analysis that classifies text reviews as **Positive** or **Negative** using three different machine learning and deep learning models.

The application supports both **single text prediction** and **batch CSV prediction**, allowing users to compare multiple sentiment analysis models through an interactive Streamlit interface.


**Features**

🔹 Single text sentiment prediction
🔹 Batch CSV sentiment prediction
🔹 Three different prediction models
  - Naive Bayes
  - Logistic Regression
  - Transformer (Hugging Face)
🔹 Confidence score visualization
🔹 Interactive and modern Streamlit dashboard
🔹 Download prediction results as CSV
🔹 Sentiment distribution visualization



 **Models Used**

| Model | Type |
|--------|------|
| Naive Bayes | Classical Machine Learning |
| Logistic Regression | Classical Machine Learning |
| Transformer | Hugging Face Transformer |


**Dataset**

The models were trained using a combined dataset of:

- IMDB Movie Reviews
- Twitter Sentiment Dataset

**Training Samples:** 60,000+ reviews

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Hugging Face Transformers
- Pandas
- Matplotlib
- Joblib


## 📸 Application Modules

### 🏠 Home
Overview of the project, model information, dataset details, and performance metrics.

### ✍️ Single Prediction
Predict sentiment for individual text with confidence score and probability visualization.

### 📂 Batch Prediction
Upload a CSV file containing a **review** column to predict sentiments for multiple reviews simultaneously.

### 📊 Model Performance
Compare the performance of all three models.



** Project Structure**

sentiment_analysis_project/
│
├── app.py
├── train_model.py
├── train_model2.py
├── download_dataset.py
├── merge_datasets.py
│
├── model.pkl
├── vectorizer.pkl
├── model_logistic.pkl
├── vectorizer_logistic.pkl
│
├── merged_dataset.csv
├── sentiment_dataset.csv
├── twitter_training.csv
│
└── README.md



## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Naive Bayes | 81.68% |
| Logistic Regression | ~85% |
| Transformer | ~94% |

