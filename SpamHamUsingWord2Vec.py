import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, keyedvectors
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Pretrained Word2Vec Model
# wv = api.load('word2vec-google-news-300')
# vec_king = wv['king']
# print(vec_king)
# print(vec_king.shape)

# Importing the dataset
messages = pd.read_csv('Datasets/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.shape)

# Data cleaning and preprocessing (Can be skipped)
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)
print(corpus[:5])

# Display all the data where corpus length < 1
print('Removed Corpus:', [[i, j, k] for i, j, k in zip(list(map(len, corpus)), corpus, messages['message']) if i < 1])

# Getting all the words of sentences
words = []
for sentence in corpus:
    sent_token = sent_tokenize(sentence)
    for sent in sent_token:
        words.append(simple_preprocess(sent))
print(words[0])
print(words[1])

# train word2vec from scratch
model = gensim.models.Word2Vec(words)

# TO get all the vocabulary
print(model.wv.index_to_key)
print(model.corpus_count)
print(model.epochs)
print(model.wv.similar_by_word('good'))
print(model.wv['good'].shape)
print(words[0])


def avg_word2vec(doc):
    # remove out-of-vocabulary words
    # sent = [word for word in doc if word in model.wv.index_to_key]
    # print(sent)

    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0)
    # [np.zeros(len(model.wv.index_to_key)), axis=0]


# apply avg_word2vec for the entire sentences
X = []
for i in tqdm(range(len(words))):
    X.append(avg_word2vec(words[i]))
print(len(X))
print(X[0])

# Independent Features
X_new = np.array(X)
print("X_new", X_new)
print('Independent Features Shape:', X_new.shape)
print(X_new[0].shape)

# Dependent Features
y = messages[list(map(lambda x: len(x) > 0, corpus))]
y = pd.get_dummies(y['label'])
y = y.iloc[:, 0].values
print('Dependent Features Shape:', y.shape)

print(X[0].reshape(1, -1).shape)

# this is the final independent features
df = pd.DataFrame()
for i in range(0, len(X)):
    df = df.append(pd.DataFrame(X[i].reshape(1, -1)), ignore_index=True)
print(df.head())

df['Output'] = y
print(df.head())

df.dropna(inplace=True)
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
