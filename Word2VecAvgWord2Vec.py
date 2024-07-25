import re
from tqdm import tqdm

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

messages = pd.read_csv('Datasets/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.head())
print(messages.shape)
print(messages['message'].loc[451])

# Data cleaning and preprocessing (Using Lemmatization)
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z0-9]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
print(corpus[0])

"""Word2Vec Implementation"""
# Pretrained Word2Vec Model
# wv = api.load('word2vec-google-news-300')
# vec_king = wv['king']
# print(vec_king)
# print(wv.most_similar('man'))
# print(wv.most_similar('king'))
# print('Cosine Similarity of man and king', wv.similarity('man', 'king'))
# print('Cosine Similarity of html and programmer', wv.similarity('html', 'programmer'))
# vec = wv['king'] - wv['man'] + wv['woman']
# print(wv.most_similar([vec]))

# Lets train Word2vec from scratch
# Getting all the words of sentences
words = []
for sentence in corpus:
    sent_token = sent_tokenize(sentence)
    for sent in sent_token:
        words.append(simple_preprocess(sent))
# print(words)

model = Word2Vec(words, window=5, min_count=2)
print('All the vocabulary of the dataset:', model.wv.index_to_key)  # All the vocabulary of the dataset
print('Total Vocabulary size:', model.corpus_count)  #
print('Number of epochs used for training:', model.epochs)
print("Similar words of 'kid'", model.wv.similar_by_word('kid'))
print("Shape of word 'kid'", model.wv['kid'].shape)
print(words[73])
print(type(model.wv.index_to_key))

"""Average Word2Vec Implementation"""

# def avg_word2vec(doc):
#     # remove out-of-vocabulary words
#     # sent = [word for word in doc if word in model.wv.index_to_key]
#     # print(sent)
#
#     return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0)
#     # or [np.zeros(len(model.wv.index_to_key))], axis=0)
#
#
# # apply for the entire sentences
# X = []
# for i in tqdm(range(len(words))):
#     X.append(avg_word2vec(words[i]))
# print(type(X))
#
# X_new = np.array(X)
# print(X_new[3])
# print(X_new.shape)

w2v_words = list(model.wv.index_to_key)
sent_vectors = []  # the avg-w2v for each sentence/review is stored in this list
for sent in tqdm(words):  # for each review/sentence
    sent_vec = np.zeros(
        100)  # as word vectors are of zero length 100, you might need to change this to 300 if you use google's w2v
    cnt_words = 0  # num of words with a valid vector in the sentence/review
    for word in sent:  # for each word in a review/sentence
        if word in w2v_words:
            vec = model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))  # to check the total rows of the matrix
print(len(sent_vectors[0]))
print(type(sent_vectors))

# Independent Features
X_new = np.array(sent_vectors)
print('Independent Features Shape:', X_new.shape)
# print(X_new[3])
print(X_new[3].shape)

# Display all the data where corpus length < 1
print('Removed Corpus:', [[i, j, k] for i, j, k in zip(list(map(len, corpus)), corpus, messages['message']) if i < 1])

# Dependent Features
y = messages[list(map(lambda x: len(x) > 0, corpus))]
y = pd.get_dummies(y['label'])
y = y.iloc[:, 0].values
print('Dependent Features Shape:', y.shape)

print(sent_vectors[0].reshape(1, -1).shape)

# this is the final independent features
df = pd.DataFrame()
for i in range(0, len(sent_vectors)):
    sent_vector_reshaped = sent_vectors[i].reshape(1, -1)
    df = pd.concat([df, pd.DataFrame(sent_vector_reshaped)], ignore_index=True)
print(df.head())

df['Output'] = y
print(df.head())

df.dropna(inplace=True)  # Dropping Null Values
# print(df.isnull().sum())

X = df.drop('Output', axis=1)
y = df['Output']
# print(X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score (AvgWord2Vec):', score)

report = classification_report(y_test, y_pred)
print(report)

"""https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"""
