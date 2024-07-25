import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('Datasets/Data.csv', encoding='ISO-8859-1')
# print(df.head())

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Removing punctuations
train_data = train.iloc[:, 2:27]  # will select all rows but 2nd to 26th column.
train_data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
# print(train_data)

# Renaming column names for ease of access
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
train_data.columns = new_index
# print(train_data.head(5))

# Converting headlines to lower case
for index in new_index:
    train_data[index] = train_data[index].str.lower()
# print(train_data.head(5))

# Converting all the headlines (Top1 to Top25) of a row into a single paragraph
headlines = []
for row in range(len(train_data.index)):
    headlines.append(' '.join(str(x) for x in train_data.iloc[row, 0:25]))
# print(headlines)
# print(headlines[3])

# Implement BAG OF WORDS (BOW)
# "ngram_range" parameter refers to the range of n-grams from the text that will be included in the bag of words
# an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams
count_vector = CountVectorizer(ngram_range=(2, 2))
train_dataset = count_vector.fit_transform(headlines)
# print(train_dataset)

# Implement RandomForest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
random_forest_classifier.fit(train_dataset, train['Label'])
# print(random_forest_classifier)

# Predict for the Test Dataset
test_transform = []
for row in range(len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = count_vector.transform(test_transform)
predictions = random_forest_classifier.predict(test_dataset)

# Check accuracy
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
accuracy = accuracy_score(test['Label'], predictions)
print(accuracy)
classification = classification_report(test['Label'], predictions)
print(classification)

class_acc = np.diag(matrix) / np.sum(matrix, axis=1)  # Calculate class-specific accuracies
print(class_acc)
