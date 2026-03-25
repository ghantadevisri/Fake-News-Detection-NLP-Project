# Fake News Detection using NLP

### By Ghanta DeviSri | B.Tech CSE Data Science | SPSU

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-NLTK-green)

## Project Overview
In today's world, fake news spreads faster than real news. 
This project builds a Machine Learning model that automatically 
detects whether a news article is **FAKE or REAL** using 
Natural Language Processing techniques.

## Dataset
- Source: Kaggle News Dataset
- Size: 6,335 news articles
- Labels: FAKE or REAL
- Features: Article title and full text

## Technologies Used
- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK
- Scikit-learn
- VADER Sentiment Analysis

## NLP Pipeline
1. Exploratory Data Analysis (EDA)
2. Text Preprocessing & Cleaning
3. Tokenization & Lemmatization
4. TF-IDF Vectorization with N-grams
5. Model Training & Comparison
6. Model Evaluation
7. Sentiment Analysis
8. Model Saving with Pickle

## Models Used
| Model | Accuracy |
|---|---|
| Logistic Regression | ~98% |
| Naive Bayes | ~95% |
| Random Forest | ~99% |

## Key Findings
- FAKE news articles are longer than REAL news
- FAKE news uses more sensational phrases
- Domain mismatch observed with Indian news headlines
- TF-IDF with N-grams improved model performance

## Project Structure
```
├── News.ipynb               # Main notebook
├── news.csv                 # Dataset
├── fake_news_model.pkl      # Saved model
└── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
```

## How To Run
1. Clone this repository
2. Install required libraries
3. Open News.ipynb in Jupyter Notebook
4. Run all cells from top to bottom
