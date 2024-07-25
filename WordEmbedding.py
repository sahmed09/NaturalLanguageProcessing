import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding

print(tf.__version__)

# Word Embedding Techniques using Embedding Layer in Keras
"""Steps of Word Embedding:
1. Sentences
2. One Hot Encoding (Vocabulary Size)
3. Padding -> Post Padding and Pre Padding (OHE)
4. OHE -> Vectors"""

# Sentences
sent = ['the glass of milk',
        'the glass of juice',
        'the cup of tea',
        'I am a good boy',
        'I am a good developer',
        'understand the meaning of words',
        'your videos are good']
print(sent)

# Vocabulary size
voc_size = 500

# One Hot Representation
one_hot_rep = [one_hot(words, voc_size) for words in sent]
print(one_hot_rep)

# Word Embedding Representation  (pad_sequences make sure the size of the sentences are same)
sent_length = 8
embedded_docs = pad_sequences(one_hot_rep, padding='pre', maxlen=sent_length)  # pre padding
print(embedded_docs)

# 10 feature dimensions
dim = 10

model = Sequential()
model.add(Embedding(voc_size, dim, input_length=sent_length))
model.compile(optimizer='adam', loss='mse')
model.summary()

print(embedded_docs[0])  # 'the glass of milk',
print(model.predict(embedded_docs[0]))
print(model.predict(embedded_docs))
print(model.predict(embedded_docs)[0])

sent = ["The world is a better place",
        "Marvel series is my favourite movie",
        "I like DC movies",
        "the cat is eating the food",
        "Tom and Jerry is my favourite movie",
        "Python is my favourite programming language"]
