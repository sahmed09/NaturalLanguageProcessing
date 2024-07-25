import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print(tf.__version__)

"""Dataset: https://www.kaggle.com/c/fake-news/data"""
# Using Word Embedding Techniques with LSTM
"""Steps:
1. Dataset
2. Independent and Dependent Features
3. Cleaning the Data i) Stemming, ii) Stopwords
4. Fix a sentence length to fix the input
5. One Hot representation, Embedding Layer
6. LSTM Neural Network"""

df = pd.read_csv('Datasets/fake-news/train.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())

# Drop Nan Values
df = df.dropna()

# Get the Independent Features
X = df.drop('label', axis=1)

# Get the Dependent features
y = df['label']

print(X.shape, y.shape)

# Vocabulary size
vocab_size = 5000

"""OneHot Representation"""
messages = X.copy()
# print(messages['title'])
print(messages['title'][0])

messages.reset_index(inplace=True)

# Dataset Preprocessing
stemmer = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)
print(corpus[0])

one_hot_rep = [one_hot(words, vocab_size) for words in corpus]
# print(one_hot_rep)
print(one_hot_rep[0])

"""Embedding Representation"""
sent_length = 20
embedded_docs = pad_sequences(one_hot_rep, padding='pre', maxlen=sent_length)
print(embedded_docs)
print(embedded_docs[0])

print(len(embedded_docs), y.shape)

X_final = np.array(embedded_docs)
y_final = np.array(y)
print(X_final.shape, y_final.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

"""Creating model"""
embedding_vector_features = 40  # features representation
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))  # Dense layer is added as it is a classification problem
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""Model Training"""
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

"""Performance Metrics And Accuracy"""
y_pred = model.predict(X_test)
# y_prediction = np.argmax(y_pred, axis=-1)  # For multi-class classification
y_prediction = (y_pred > 0.5).astype("int32")  # for binary classification
matrix = confusion_matrix(y_test, y_prediction)
scores = accuracy_score(y_test, y_prediction)
print(matrix)
print(scores)

"""Adding Dropout"""
# Creating model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Dense layer is added as it is a classification problem
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Performance Metrics And Accuracy
y_pred = model.predict(X_test)
# y_prediction = np.argmax(y_pred, axis=-1)  # For multi-class classification
# y_pred = np.where(y_pred > 0.6, 1, 0)  # AUC ROC Curve
y_prediction = (y_pred > 0.5).astype("int32")  # for binary classification
# print(y_prediction)

matrix = confusion_matrix(y_test, y_prediction)
print(matrix)

scores = accuracy_score(y_test, y_prediction)
print('Accuracy:', scores)

report = classification_report(y_test, y_prediction)
print(report)
