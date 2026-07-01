# Stock Market Sentiment Analysis

Predicting stock price movements using NLP sentiment analysis on financial news headlines.

## Project Overview
This project builds a two-component pipeline:
- Component A: Classifies financial news headlines as positive, negative, or neutral using TF-IDF + XGBoost, benchmarked against FinBERT
- Component B: Predicts AAPL stock price movement using technical indicators (MA5, MA10, volatility, returns)

## Tech Stack
Python, FinBERT (HuggingFace), XGBoost, yfinance, Scikit-learn, Pandas, Matplotlib, Streamlit

## Results
| Model | Task | Accuracy |
|---|---|---|
| FinBERT | Sentiment Classification | 88.98% |
| TF-IDF + XGBoost | Sentiment Classification | 74.28% |
| XGBoost | Price Movement Prediction | 52.00% |

## Progress
- [x] Week 1: Data collection & exploration
- [x] Week 2: Sentiment scoring with FinBERT
- [x] Week 3: ML model training
- [x] Week 4: Streamlit dashboard

## Limitations
- Financial PhraseBank has no timestamps, so sentiment data cannot be directly merged with stock price data. Both components are demonstrated separately.
- The custom XGBoost sentiment model underperforms on negative class due to class imbalance (only 12% negative samples).
- 52% price prediction accuracy is consistent with the Efficient Market Hypothesis.

## Dataset
Financial PhraseBank — [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

## How to Run
streamlit run app.py
