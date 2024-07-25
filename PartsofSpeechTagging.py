from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy

"""Implementation of Parts-of-Speech tagging using NLTK"""
print('Implementation of Parts-of-Speech tagging using NLTK')
text = "NLTK is a powerful library for natural language processing."
words = word_tokenize(text)

pos_tags = pos_tag(words)

print("Original Text:", text)
print("PoS Tagging Result:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")

"""Implementation of Parts-of-Speech tagging using Spacy"""
print('\nImplementation of Parts-of-Speech tagging using Spacy')
nlp = spacy.load("en_core_web_sm")

text = "SpaCy is a popular natural language processing library."
doc = nlp(text)
print("Original Text: ", text)
print("PoS Tagging Result:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
