import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

paragraph = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""
# paragraph = """This movie is very scary and long. This movie is not scary and is slow. This movie is spooky and good."""

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
print(len(corpus))

# Creating the Bag of Words (BOW) model
cv = CountVectorizer(binary=False, ngram_range=(1, 1))  # binary=True for creating Binary Bag of Words
X = cv.fit_transform(corpus).toarray()

print(cv.vocabulary_)  # dictionary contains word index mapping
print('Unique Word List:', cv.get_feature_names_out())
print('Bag of Words Matrix:\n', X)
print(X.shape)
print(X[0])

# create a dataframe and show the result visually
df = pd.DataFrame(data=X, columns=cv.get_feature_names_out(), index=corpus)
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

# Creating the Bag of Words (BOW) model
count_vector = CountVectorizer(max_features=2500, binary=True)
X = count_vector.fit_transform(corpus).toarray()
print(X.shape)
print(X[1])

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score (Bog of Words):', score)

report = classification_report(y_test, y_pred)
print(report)

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

# Display all the data where corpus length < 1
print('Removed Corpus:', [[i, j, k] for i, j, k in zip(list(map(len, corpus)), corpus, messages['message']) if i < 1])

# Creating the Bag of Words (BoW) model
count_vector = CountVectorizer(max_features=2500, binary=True)
X = count_vector.fit_transform(corpus).toarray()
print(X.shape)
print(X[1])

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy Score (Bog of Words):', score)

report = classification_report(y_test, y_pred)
print(report)
