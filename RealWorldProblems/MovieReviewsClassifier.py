import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = ["kolkata big city india trade", "mumbai financial capital india", "delhi capital india",
        "kolkata capital colonial times",
        "bangalore tech hub india software", "mumbai hub trade commerce stock exchange", "kolkata victoria memorial",
        "delhi india gate",
        "mumbai gate way india trade business", "delhi red fort india", "kolkata metro oldest india",
        "delhi metro largest metro network india"]

# using the count vectorizer
count = CountVectorizer()
word_count = count.fit_transform(text)
# print(word_count)
print(word_count.shape)
print(word_count.toarray())

# TF-IDF transformation
tfidf_transformer = TfidfTransformer()
tf_idf_vector = tfidf_transformer.fit_transform(word_count)
feature_names = count.get_feature_names_out()
first_document_vector = tf_idf_vector[1]
df_tfidf = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=['tfidf'])
print(df_tfidf.sort_values(by=['tfidf'], ascending=False))

"""Movie Reviews Classifier using TF-IDF"""
print('\nMovie Reviews Classifier using TF-IDF')

train_csv = pd.read_csv('../Datasets/imdb-movie-reviews-dataset/train_data.csv')
test_csv = pd.read_csv('../Datasets/imdb-movie-reviews-dataset/test_data.csv')
print(train_csv.head())

# stopword removal and lemmatization
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

train_X_non = train_csv['0']  # '0' refers to the review text
train_y = train_csv['1']  # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test_csv['0']
test_y = test_csv['1']

train_X = []
test_X = []

# text pre processing
for i in range(len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stop_words)]
    review = ' '.join(review)
    train_X.append(review)

for i in range(len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stop_words)]
    review = ' '.join(review)
    test_X.append(review)

print(train_X[10])

# tf-idf
tf_idf = TfidfVectorizer()

# applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)
print("n_samples: %d, n_features: %d" % X_train_tf.shape)

# transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)
print("n_samples: %d, n_features: %d" % X_test_tf.shape)

# Naive Bayes Classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)
y_pred = naive_bayes_classifier.predict(X_test_tf)

accuracy = metrics.accuracy_score(test_y, y_pred)
print('Accuracy Score:', accuracy)

c_matrix = metrics.confusion_matrix(test_y, y_pred)
print(c_matrix)

report = metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative'])
print(report)

# Test Prediction on Reviews Classifier Using TF-IDF
test = [
    "This is unlike any kind of adventure movie my eyes have ever seen in such a long time, the characters, the musical score for every scene, the story, the beauty of the landscapes of Pandora, the rich variety and uniqueness of the flora and fauna of Pandora, the ways and cultures and language of the natives of Pandora, everything about this movie I am beyond impressed and truly captivated by. Sam Worthington is by far my favorite actor in this movie along with his character Jake Sulley, just as he was a very inspiring actor in The Shack Sam Worthington once again makes an unbelievable mark in one of the greatest and most captivating movies you'll ever see. "]

review = re.sub('[^a-zA-Z]', ' ', test[0])
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if word not in set(stop_words)]
test_processed = [' '.join(review)]
print(test_processed)

test_input = tf_idf.transform(test_processed)
print(test_input.shape)

# res = naive_bayes_classifier.predict(test_input)[0]
res = naive_bayes_classifier.predict(test_input)
print(res, res[0])

# 0 = bad review, 1 = good review
if res[0] == 1:
    print('Good Review')
elif res[0] == 0:
    print('Bad Review')
