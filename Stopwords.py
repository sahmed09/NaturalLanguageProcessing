import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

print(stopwords.words('english'))

"""Removing stop words with NLTK"""
print('Removing stop words with NLTK')

example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
filtered_sentence = [word for word in word_tokens if not word.lower() in stop_words]

filtered_sentences = []
for word in word_tokens:
    if word not in stop_words:
        filtered_sentences.append(word)
print(word_tokens)
print(filtered_sentence)

new_filtered_words = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
new_clean_text = ' '.join(new_filtered_words)
print(new_clean_text)

"""Removing stop words with SpaCy"""
print('\nRemoving stop words with SpaCy')
nlp = spacy.load('en_core_web_sm')

text = "There is a pen on the table"
doc = nlp(text)
filtered_words = [token.text for token in doc if not token.is_stop]  # Remove stopwords
clean_text = ' '.join(filtered_words)

print("Original Text:", text)
print("Text after Stopword Removal:", clean_text)

"""Removing stop words with Genism"""
print('\nRemoving stop words with Genism')
new_text = "The majestic mountains provide a breathtaking view."

new_filtered_text = remove_stopwords(new_text)

print("Original Text:", new_text)
print("Text after Stopword Removal:", new_filtered_text)

"""Removing stop words with SkLearn"""
print('\nRemoving stop words with SkLearn')
new_text = "The quick brown fox jumps over the lazy dog."
new_words = word_tokenize(new_text)
