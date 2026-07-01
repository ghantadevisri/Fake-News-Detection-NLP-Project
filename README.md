# Fake News Detection using NLP

### By Ghanta DeviSri | B.Tech CSE Data Science | SPSU

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-NLTK-green)

## Project Overview
In today's world, fake news spreads faster than real news,
especially on social media and online platforms. This project
builds a Machine Learning model that automatically detects
whether a news article is FAKE or REAL using Natural
Language Processing techniques.

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
- WordCloud

## NLP Pipeline
1. Exploratory Data Analysis (EDA)
2. Text Preprocessing and Cleaning
3. Tokenization and Lemmatization
4. TF-IDF Vectorization with N-grams
5. Model Training and Comparison
6. Model Evaluation (Confusion Matrix, F1 Score)
7. Sentiment Analysis
8. Model Saving with Pickle

## Models and Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 91.71% |
| Naive Bayes | 88.63% |
| Random Forest | 91.79% |

Best performing model: Random Forest with 91.79% accuracy

## Key Visualizations
- FAKE vs REAL news distribution chart
- Article length comparison box plot
- Word clouds for FAKE and REAL news
- Top 20 most frequent words
- Bigram phrase word cloud
- Sentiment distribution chart
- Confusion matrix heatmap
- Model performance comparison chart

## Key Findings
- FAKE news articles are longer than REAL news articles
- FAKE news commonly uses sensational phrases and
  health misinformation
- Trump was the most common word in both FAKE and
  REAL news (dataset is from 2016-2017)
- Domain mismatch observed when tested on Indian
  news headlines
- TF-IDF with N-grams improved model performance
- FAKE news tends to have more negative sentiment
  than REAL news

## Project Structure
```
├── News.ipynb                  # Main Jupyter notebook
├── news.csv                    # Dataset
├── fake_news_model.pkl         # Saved trained model
└── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
```


## How To Run

### 1. Clone this repository
git clone https://github.com/ghantadevisri/Fake-News-Detection-NLP-Project.git

### 2. Install required libraries
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud vaderSentiment

### 3. Download NLTK data
Open Python and run:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

### 4. Place dataset in project folder
Make sure news.csv is in the same folder as News.ipynb

### 5. Open Jupyter Notebook
jupyter notebook

### 6. Open News.ipynb

### 7. Run all cells from top to bottom
Press Shift + Enter on each cell, or click
Kernel -> Restart & Run All

## Note
This model was trained on American news data from 2016-2017.
It may show domain mismatch when tested on Indian news headlines.
