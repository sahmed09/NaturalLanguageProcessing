from textblob import TextBlob

"""TextBlob: Simplified Text Processing
TextBlob is a Python library for processing textual data. It provides a simple API for diving into common natural
language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis,
classification, and more."""

"""Spelling Correction"""
b = TextBlob("I havv goood speling!")
print(b.correct())

text = """
The titular threat of The Blob has alays struck me as the ultiate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilaing flsh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo ssenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
"""
b = TextBlob(text)
print(b.correct())

"""Get Word and Noun Phrase Frequencies"""
monty = TextBlob("We are no longer the Knights who say Ni. "
                 "We are now the Knights who say Ekki ekki ekki PTANG.")
print(monty.word_counts['ekki'])
