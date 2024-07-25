import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(tf.__version__)

df = pd.read_csv('Datasets/fake-news/train.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())

df = df.dropna()  # Dropping Nan Values
print(df.shape)

# Ged the Independent Features
X = df.drop('label', axis=1)

# Ged the Dependent Features
y = df['label']

print(X.shape, y.shape)

# Check whether dataset is balanced or not
print(y.value_counts())

# vocabulary size
vocab_size = 5000

"""One Hot Representation"""
messages = X.copy()
print(messages['title'][0])
print(messages.head())

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
print(corpus[0])

one_hot_rep = [one_hot(words, vocab_size) for words in corpus]
print(one_hot_rep[0])

"""Embedding Representation"""
sent_length = 20
embedded_docs = pad_sequences(one_hot_rep, padding='pre', maxlen=sent_length)
print(embedded_docs)
print(embedded_docs[0])

# Creating Model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_final = np.array(embedded_docs)
y_final = np.array(y)
print(X_final.shape, y_final.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

# Model Training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

y_prediction = model.predict(X_test)
y_pred = np.where(y_prediction >= 0.5, 1, 0)  # AOC RUC Curve
# y_pred = (y_prediction > 0.5).astype('int32')

score = accuracy_score(y_test, y_pred)
print('Accuracy:', score)

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)

report = classification_report(y_test, y_pred)
print(report)

"""IMDB Datasets"""
n_unique_words = 10000  # cut texts after this number of words
maxlen = 200
batch_size = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=12, validation_data=[x_test, y_test])
print(history.history['loss'])
print(history.history['accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('model loss vs accuracy')
plt.xlabel('epoch')
plt.legend(['loss', 'accuracy'], loc='upper right')
plt.show()
