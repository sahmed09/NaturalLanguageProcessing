import re
import nltk
from gensim.utils import tokenize
# from gensim.summarization.textcleaner import split_sentences
from spacy.lang.en import English
from keras.preprocessing.text import text_to_word_sequence

# nltk.download()

paragraph = """Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences, called tokens. If the text is split into words, then it’s called as 'Word Tokenization' and if it's split into sentences then it’s called as 'Sentence Tokenization'. Generally 'space' is used to perform the word tokenization and characters like 'periods, exclamation point and newline char are used for Sentence Tokenization."""

"""Tokenization Using NLTK"""
print('Tokenization Using NLTK')

# Tokenizing words
words = nltk.word_tokenize(paragraph)
print(words)

# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
print(sentences)

text = """There are multiple ways we can perform tokenization on given text data. We can choose any method based on langauge, library and purpose of modeling."""

"""Tokenization Using Python's Inbuilt Method"""
print("\nTokenization Using Python's Inbuilt Method")

# Word Tokenization
tokens = text.split()  # Split text by whitespace
print(tokens)

# Sentence Tokenization
tokens = text.split('. ')  # split the given text by full stop (.)
print(tokens)

"""Tokenization Using Regular Expressions (RegEx)"""
print('\nTokenization Using Regular Expressions (RegEx)')

# Word Tokenization
tokens = re.findall('[\w]+', text)
print(tokens)

# Sentence Tokenization
tokens_sent = re.compile('[.!?] ').split(text)
print(tokens_sent)

"""Tokenization Using spaCy"""
print('\nTokenization Using spaCy')

# Word Tokenization
nlp = English()  # Load English tokenizer
my_doc = nlp(text)

token_list = []
for token in my_doc:
    token_list.append(token.text)
print(token_list)

# Sentence Tokenization
nlp = English()  # Load English tokenizer
sbd = nlp.create_pipe('sentencizer')  # Create the pipeline 'sentencizer' component
nlp.add_pipe(sbd)  # Add component to the pipeline

doc = nlp(text)
sentence_list = []
for sentence in doc.sents:
    sentence_list.append(sentence.text)
print(sentence_list)

"""Tokenization using Keras"""
print('\nTokenization using Keras')

# Word Tokenization
tokens = text_to_word_sequence(text)
print(tokens)

# Sentence Tokenization
tokens = text_to_word_sequence(text, split=".", filters="!.\n")
print(tokens)
tokens = text_to_word_sequence(text, split=".")
print(tokens)

"""Tokenization using Gensim"""
print('\nTokenization using Gensim')

# Word Tokenization
tokens = list(tokenize(text))
print(tokens)

# Sentence Tokenization
# sentences = list(split_sentences(text))
# print(sentences)
