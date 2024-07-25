import re
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Custom way of plotting confusion matrix. While using just specify the class names and cm (confusion matrix)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


"""Dataset: https://www.kaggle.com/c/fake-news/data"""

news = pd.read_csv('Datasets/fake-news/train.csv')
print(news.head())

# Get the Independent Features
X = news.drop('label', axis=1)
print(X.head)

# Get the Dependent Features
y = news['label']
print(y.head())
print(news.shape)

# Drop nan Values
news = news.dropna()
print(news.shape)
print(news.head(10))

# Copy the original df
messages = news.copy()
print(messages.head(10))

# Reset index as after using dropna(), some indexes will be missing,
# The reset_index() method allows reset the index back to the default 0, 1, 2 etc indexes.
messages.reset_index(inplace=True)
print(messages.head(10))
print(messages['text'][6])

# Preprocessing the data
stemmer = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
print(corpus[1])

# Applying TfidfVectorizer
# Creating the TF-IDF model
tf_idf_vector = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))  # (1, 3) take inputs of 1/2/3 words as a feature
X = tf_idf_vector.fit_transform(corpus).toarray()
print(X.shape)

y = messages['label']
print(y.head())

# Divide the dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Get output feature names for transformation and Get parameters for this estimator.
print(tf_idf_vector.get_feature_names_out()[:20])
print(tf_idf_vector.get_params())

# Show how training dataset looks like
count_df = pd.DataFrame(X_train, columns=tf_idf_vector.get_feature_names_out())
print(count_df.head())

# MultinomialNB Algorithm -> Works better with text data.
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(X_train, y_train)
y_prediction = multinomial_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
print("accuracy:   %0.3f" % accuracy)

confusion_m = confusion_matrix(y_test, y_prediction)
print(confusion_m)

plot_confusion_matrix(confusion_m, classes=['FAKE', 'REAL'])
print(y_train.shape)

# Passive Aggressive Classifier Algorithm -> Also works better with text data.
linear_clf = PassiveAggressiveClassifier(max_iter=50)
linear_clf.fit(X_train, y_train)
y_prediction = linear_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print("accuracy:   %0.3f" % accuracy)

confusion_m = confusion_matrix(y_test, y_prediction)
plot_confusion_matrix(confusion_m, classes=['FAKE Data', 'REAL Data'])

# Multinomial Classifier with Hyperparameter
classifier = MultinomialNB(alpha=0.1)
previous_score = 0
for alpha in np.arange(0, 1, 0.1):
    sub_classifier = MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train, y_train)
    y_pred = sub_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    if accuracy > previous_score:
        classifier = sub_classifier
    print("Alpha: {}, Score : {}".format(alpha, accuracy))

# Get Features names
feature_names = tf_idf_vector.get_feature_names_out()
print(classifier.feature_log_prob_[0])

# Most real
print(sorted(zip(classifier.feature_log_prob_[0], feature_names), reverse=True)[:20])

# Most fake
print(sorted(zip(classifier.feature_log_prob_[0], feature_names))[:100])
