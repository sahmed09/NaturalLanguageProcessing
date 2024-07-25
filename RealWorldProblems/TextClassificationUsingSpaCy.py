import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

df_amazon = pd.read_csv('../Datasets/amazon_alexa.tsv', sep='\t')
print(f'Shape of data: {df_amazon.shape}')
print(df_amazon.head())
print(df_amazon.info)
print(df_amazon.feedback.value_counts())

# Tokenizing the Text
# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stop words
# nlp = spacy.load('en')
stop_words = STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vector
parser = English()


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    """This function will accepts a sentence as input and processes the sentence into tokens, performing lemmatization,
        lowercasing, removing stop words and punctuations."""
    # Creating our token object which is used to create documents with linguistic annotations
    my_tokens = parser(sentence)

    # lemmatizing each token and converting each token in lower case
    # Note that spaCy uses '-PRON-' as lemma for all personal pronouns like me, I etc
    my_tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in my_tokens]

    # Removing stop words
    my_tokens = [word for word in my_tokens if word not in stop_words and word not in punctuations]

    # Return preprocessed list of tokens
    return my_tokens


# Data Cleaning
# Custom transformer using spaCy
class Predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        """Override the transform method to clean text"""
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Basic function to clean the text
def clean_text(text):
    """Removing spaces and converting the text into lowercase"""
    if isinstance(text, str):
        return text.strip().lower()
    else:
        # Handle non-string values (e.g., return empty string or raise an error)
        return ""  # Replace with your preferred handling (e.g., logging or error handling)


# Feature Engineering
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tf_idf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

# Create Train and Test Datasets
X = df_amazon['verified_reviews']  # The features we want to analyse
y = df_amazon['feedback']  # The labels, in this case feedback

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f'X_train dimension: {X_train.shape}')
print(f'y_train dimension: {y_train.shape}')
print(f'X_test dimension: {X_test.shape}')
print(f'y_train dimension: {y_test.shape}')

# Creating a Pipeline and Generating the Model¶
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([('cleaner', Predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])
pipe.fit(X_train, y_train)

predicted = pipe.predict(X_test)

print(f'Logistic Regression Accuracy: {metrics.accuracy_score(y_test, predicted)}')
print(f'Logistic Regression Precision: {metrics.precision_score(y_test, predicted)}')
print(f'Logistic Regression Recall: {metrics.recall_score(y_test, predicted)}')

c_matrix = metrics.confusion_matrix(y_test, predicted)
print(c_matrix)

report = metrics.classification_report(y_test, predicted, target_names=['Negative', 'Positive'])
print(report)
