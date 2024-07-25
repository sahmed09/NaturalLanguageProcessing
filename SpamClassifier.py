import pandas as pd

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
emails = pd.read_csv('Datasets/smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
print(emails.shape)

# Data cleaning and preprocessing
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(emails)):
    review = re.sub('[^a-zA-Z]', ' ', emails['message'][i])
    review = review.lower()
    review = review.split()
    # review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]  # Stemming works better
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus[:5])

# Creating the Bag of Words Model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
print(X)
print(X.shape)

# # Creating the TF-IDF model
# cv = TfidfVectorizer(max_features=2500)
# X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(emails['label'])  # coverts the label (ham/spam) into dummy variables (true/false)
y = y.iloc[:, 1].values  # Converts it into a single value column (if false then it is ham, if true then it is spam)
print(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# random_state=0/random_state=42 meaning we get the same train and test sets across different executions

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_prediction = spam_detect_model.predict(X_test)
print(y_prediction)

# Compare y_test and y_prediction
confusion_m = confusion_matrix(y_test, y_prediction)
print(confusion_m)

# Check Accuracy
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
