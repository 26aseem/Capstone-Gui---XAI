# Import the required packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import streamlit as st
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import string
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
# Set Recursion Limit
import sys
sys.setrecursionlimit(40000)
import re  
import nltk  
import regex as re
nltk.download('stopwords')  
from nltk.corpus import stopwords  
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import streamlit.components.v1 as components

### XAI - Explainable Artificial Intelligence
# Dataset
# Load Dataset
tweets = pd.read_csv("dataset.csv",lineterminator='\n')[["text","type","dataset\r"]]
tweets.columns = ["text","type", "dataset"]
tweets['dataset'] = tweets['dataset'].str.replace(r'\r', '')
# Selection of Input & Output Variables
X = tweets.loc[:, 'text']
Y = tweets.loc[:, 'type']
X = list(X)


def preprocess_dataset(d):    
    # Define count variables
    cnt=0
    punctuation_count = 0
    digit_count = 0
        
    # Convert the corpus to lowercase
    lower_corpus = []
    for i in range(len(d)):
        lower_corpus.append(" ".join([word.lower() for word in d[i].split()]))
                    
    # Remove any special symbol or punctuation
    without_punctuation_corpus = []
    for i in range(len(lower_corpus)):
        p = []
        for ch in lower_corpus[i]:
            if ch not in string.punctuation:
                p.append(ch)
            else:
                p.append(" ")
                # Count of punctuation marks removed
                punctuation_count += 1
        x = ''.join(p)
        if len(x) > 0:  
            without_punctuation_corpus.append(x)
      
    # Remove urls with http, https or www and Retweets RT
    without_url_corpus = []
    for i in range(len(without_punctuation_corpus)):
        text = without_punctuation_corpus[i]
        text = re.sub(r"http\S*||www\S*", "", text)
        text = re.sub(r"RT ", "", text)
        without_url_corpus.append(text)
        
    # Remove special characters and numbers from the corpus
    without_digit_corpus = []
    for i in range(len(without_url_corpus)):
        p = []
        for word in without_url_corpus[i].split():
            if word.isalpha():
                p.append(word)
            else:
                # Count of punctuation marks removed
                digit_count += 1
        x = ' '.join(p)
        without_digit_corpus.append(x)
        
            
    # Tokenize the corpus
    # word_tokenize(s): Tokenize a string to split off punctuation other than periods
    # With the help of nltk.tokenize.word_tokenize() method, we are able to extract the tokens
    # from string of characters by using tokenize.word_tokenize() method. 
    # Tokenization was done to support efficient removal of stopwords
    total_count = 0
    tokenized_corpus = []
    for i in without_digit_corpus:
        tokenized_tweet = nltk.word_tokenize(i)
        tokenized_corpus.append(tokenized_tweet)
        # Count the length of tokenized corpus
        total_count += len(list(tokenized_tweet))
    
    
    # Remove Stopwords
    stopw = stopwords.words('english')
    count = 0
    tokenized_corpus_no_stopwords = []  
    for i,c in enumerate(tokenized_corpus): 
        tokenized_corpus_no_stopwords.append([])
        for word in c: 
            if word not in stopw:  
                tokenized_corpus_no_stopwords[i].append(word) 
            else:
                count += 1

    # lemmatization and removing words that are too large and small
    lemmatized_corpus = []
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    ct = 0
    cnt_final=0
    dictt = {}
    for i in range(0,len(tokenized_corpus_no_stopwords)):
        lemmatized_corpus.append([])
        for w in tokenized_corpus_no_stopwords[i]:
            # lematizing only those words whose length >= 2 and <=10
            # Considering words with length greater than or equal to 2 and less than or equal to 10
            if(len(w)>2 and len(w)<=10):
                lemmatized_corpus[i].append(lemmatizer.lemmatize(w))
                cnt_final+=1
            # Count of final corpus
            # This is the length of total corpus that went through the process of lematization
            ct+=1
  
    ############## Removing words of large and small length
    # Doing a survey to find out the length of words so we can remove the too small and too large words from the Corpus
    # plt.bar(*zip(*dictt.items()))
    # plt.show()

    # Punctuation Preprocessing
    preprocessed_corpus = []
    for i,c in enumerate(lemmatized_corpus):
        preprocessed_corpus.append([])
        for word in c:
            x = ''.join([ch for ch in word if ch not in string.punctuation])
            if len(x) > 0:
                preprocessed_corpus[i].append(x)
        
   
    # Clear unwanted data variables to save RAM due to memory limitations
    del lower_corpus
    del without_punctuation_corpus
    del without_digit_corpus
    del tokenized_corpus
    del tokenized_corpus_no_stopwords
    del lemmatized_corpus
    return preprocessed_corpus

# Preprocess the Input Variables
preprocessed_corpus = preprocess_dataset(X)

data_corpus = []
for i in preprocessed_corpus:
    data_corpus.append(" ".join([w for w in i]))
    
# Vectorization of Preprocessed Tweets

tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words='english',min_df=2, max_features=50, ngram_range = (1,3))
X_tfidf = tfidfvectorizer.fit_transform(data_corpus)

# Feature extraction
feature_names = tfidfvectorizer.get_feature_names()

# Splitting the input into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf.toarray(), Y, train_size = 0.7)

# intializing the model
model = LGBMClassifier()
model.fit(X_train,Y_train)

# Instantiating the explainer object by passing in the training set, and the extracted features
explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names=feature_names,verbose=True, mode='classification', class_names=[0,1,2])
# Streamlit Code starts here
st.title('XAI - Explainable Artificial Intelligence')
st.markdown("The dashboard will help the users verify the efficiency of the classification model used here")
#st.sidebar.title("Select the Tweet from the database for XAI")


h = st.slider('Select the Tweet using the slider', 0, len(X)-1, 18)

idx=0 # the rows of the dataset
explainable_exp = explainer_lime.explain_instance(X_tfidf.toarray()[h], model.predict_proba, num_features=10, labels=[0,1,2])
#explainable_exp.show_in_notebook(show_table=True, show_all=False)
html = explainable_exp.as_html()


st.write('**Tweet:**', X[h])
st.write('**Label:**', Y[h])

components.html(html, height=800)