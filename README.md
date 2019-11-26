# SentimentAnalysis
Sentiment Analysis - Movie Reviews from IMDB Dataset - LSTM &amp; CNN



## Introduction
- **Problem:** Identify positive or negative opinions from movie reviews
- **Purpose:** The outcome can be integrated into the recommender system to tackle data sparsity problems and further improve precision
- **Data Source:** 50,000 records from iMDB [(Kaggle)](https://www.kaggle.com/utathya/imdb-review-dataset)



## Analytic Process Framework
### 1. Data Exploration: 
 - Check the number of records and the length of reviews for each category
 - Use seaborn to visualise data

### 2. Feture Engineering - Data Cleaning: 
- remove HTML line break tag: <br />
- convert words to lower case
- lemmatization
- tokenizer-encode text as a sequence of word indexes
- *Didn't remove stopwords: to avoid the meaning of sentence changes (e.g. I didn't like it -> I like it -> positive?)
- pad_sequences 

### 3. Modelling:
- LSTM Model
- CNN Model



## Environment
- Language : Python 3
- Libraries: Pandas, seaborn, keras, nltk
