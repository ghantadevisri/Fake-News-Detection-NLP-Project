import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

st.set_page_config(page_title="Stock Sentiment Analysis", page_icon="📈", layout="wide")

st.title("📈 Stock Market Sentiment Analyzer")
st.markdown("Analyze financial news headlines using FinBERT + XGBoost")

# Sidebar
st.sidebar.header("About This Project")
st.sidebar.markdown("""
- **Dataset**: Financial PhraseBank (4837 headlines)
- **Sentiment Model**: TF-IDF + XGBoost (74.28% accuracy)
- **FinBERT Accuracy**: 88.98%
- **Stock**: AAPL (2020-2024)
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\gdevi\OneDrive\Documents\Devi's\projects\financial_dataset_clean.csv")
    stock = pd.read_csv(r"C:\Users\gdevi\OneDrive\Documents\Devi's\projects\aapl_stock_prices.csv", index_col="Date", parse_dates=True)
    return df, stock

df, stock = load_data()

# Train model on load
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    le = LabelEncoder()
    X = vectorizer.fit_transform(df["headline"])
    y = le.fit_transform(df["sentiment"])
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
    model.fit(X, y)
    return model, vectorizer, le

model, vectorizer, le = train_model(df)

# Section 1 - Sentiment Predictor
st.header("🔍 Predict Headline Sentiment")
user_input = st.text_area("Enter a financial news headline:", 
                           placeholder="e.g. Company profits surge to record highs...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a headline!")
    else:
        X_input = vectorizer.transform([user_input])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        label = le.inverse_transform([pred])[0]
        confidence = round(float(proba.max()) * 100, 2)

        if label == "positive":
            st.success(f"✅ Sentiment: **POSITIVE** (Confidence: {confidence}%)")
        elif label == "negative":
            st.error(f"🔴 Sentiment: **NEGATIVE** (Confidence: {confidence}%)")
        else:
            st.info(f"⚪ Sentiment: **NEUTRAL** (Confidence: {confidence}%)")

# Section 2 - Dataset Overview
st.header("📊 Dataset Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["sentiment"].value_counts().plot(kind="bar", color=["#95a5a6", "#2ecc71", "#e74c3c"], edgecolor="black", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title("Original Labels")
    st.pyplot(fig)

with col2:
    st.subheader("FinBERT vs Original Labels")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    comparison = pd.DataFrame({
        "Original": df["sentiment"].value_counts(),
        "FinBERT": df["finbert_sentiment"].value_counts()
    })
    comparison.plot(kind="bar", ax=ax2, edgecolor="black")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_title("Label Comparison")
    st.pyplot(fig2)

# Section 3 - AAPL Stock Price
st.header("📈 AAPL Stock Price (2020-2024)")
fig3, ax3 = plt.subplots(figsize=(12, 4))
stock["Close"].plot(ax=ax3, color="#3498db", linewidth=1.5)
ax3.set_title("AAPL Closing Price")
ax3.set_ylabel("Price (USD)")
st.pyplot(fig3)

# Section 4 - Model Performance
st.header("🤖 Model Performance Summary")
col3, col4 = st.columns(2)
with col3:
    st.metric("FinBERT Accuracy", "88.98%", "Industry-standard model")
with col4:
    st.metric("XGBoost Accuracy", "74.28%", "Custom trained model")

st.info("💡 Price movement predictor achieved 52% accuracy — consistent with the Efficient Market Hypothesis, which suggests short-term price direction is difficult to predict using technical indicators alone.")