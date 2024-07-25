import re

import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec, keyedvectors
import gensim.downloader as api
from wordcloud import WordCloud
import matplotlib.pyplot as plt

paragraph = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
text = re.sub(r'\s+', ' ', text)  # removes one or more whitespace characters
text = text.lower()
text = re.sub(r'\d', ' ', text)  # removes decimal digit character
text = re.sub(r'\s+', ' ', text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
print(sentences)

# Training the Word2Vec model
model = Word2Vec(sentences=sentences,
                 vector_size=100,  # Dimensionality of the word vectors
                 window=5,  # Maximum distance between the current and predicted word within a sentence
                 sg=0,  # Skip-Gram model (1 for Skip-Gram, 0 for CBOW)
                 min_count=1,  # Ignores all words with a total frequency lower than this
                 workers=4)  # Number of CPU cores to use for training the model

words = model.wv.key_to_index  # All the vocabulary of the dataset
print(words)

word = model.wv.key_to_index['sentences']  # for looking up a specific key's integer index
print("Index of 'sentences':", word)

# Finding Word Vectors
vector = model.wv['splitting']
print("Vector representation of 'splitting'\n:", vector)
print("Vector shape of 'splitting':", vector.shape)

# Most Similar Words
similar = model.wv.most_similar('text')
print("Most similar words of 'text'", similar)

# # Using Google's pretrained model 'word2vec-google-news-300' (Pretrained on google news text data on more than
# # 3 billion words and it has 300 dimensions as output
# wv = api.load('word2vec-google-news-300')
# vec_king = wv['king']
# print(vec_king)
# print(wv.most_similar('man'))
# print(wv.most_similar('king'))
# print('Cosine Similarity of man and king', wv.similarity('man', 'king'))
# print('Cosine Similarity of html and programmer', wv.similarity('html', 'programmer'))
# vec = wv['king'] - wv['man'] + wv['woman']
# print(wv.most_similar([vec]))

# WordCloud (Not Related to Word2Vec)
text = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""

# Generate a word cloud image
word_cloud = WordCloud().generate(text)

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# lower max_font size
word_cloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
