import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

paragraph = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""

# Cleaning the texts
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    # review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

# Creating the TF-IDF model
tf_idf = TfidfVectorizer(ngram_range=(1, 1), max_features=None)
X = tf_idf.fit_transform(corpus).toarray()

print(tf_idf.vocabulary_)  # dictionary contains word index mapping
print('Feature Names/Unique Word List:', tf_idf.get_feature_names_out())
print('TF-IDF Matrix:\n', X)
print(X.shape)
print(X[0])

# create a dataframe and show the result visually
df = pd.DataFrame(data=X, columns=tf_idf.get_feature_names_out(), index=corpus)
print(df)

"""SMSSpamCollection"""
print('\nSMSSpamCollection')
messages = pd.read_csv('Datasets/smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.head())
print(messages.shape)
print(messages['message'].loc[451])

# Data cleaning and preprocessing (Using Stemming)
stemmer = PorterStemmer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z0-9]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)
print(corpus[0])

# Creating the TFIDF model
X = corpus

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

tf_idf = TfidfVectorizer(max_features=2500)
X_train = tf_idf.fit_transform(X_train).toarray()
X_test = tf_idf.transform(X_test).toarray()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score (TF-IDF):', score)

report = classification_report(y_test, y_pred)
print(report)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score (TF-IDF):', score)

report = classification_report(y_test, y_pred)
print(report)