import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# print(stopwords.words('english'))
# print(stopwords.words('bengali'))

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)

print(sentences)

print(stemmer.stem('finalized'))
print(stemmer.stem('finally'))
print(stemmer.stem('final'))
