import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

"""Language Translation using sequence-to-sequence learning (Encoder-Decoder)"""

batch_size = 64  # Batch size for training
epochs = 100  # Number of epochs to train for
latent_dim = 256  # Latent dimensionality of the encoding space
num_samples = 10000  # Number of samples to train on

data_path = 'Datasets/fra.txt'  # Dataset path

# Vectorize the data
input_texts = []  # English text
target_texts = []  # French text
input_characters = set()  # English words
target_characters = set()  # French words
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

print(input_texts)
print(target_texts)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_sequence_length = max([len(txt) for txt in input_texts])
max_decoder_sequence_length = max([len(txt) for txt in target_texts])

print(input_characters)
print(target_characters)

print('Number of Samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique decoder tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_sequence_length)
print('Max sequence length for outputs:', max_decoder_sequence_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
print(input_token_index)
print(target_token_index)

encoder_input_data = np.zeros((len(input_texts), max_encoder_sequence_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_sequence_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_sequence_length, num_decoder_tokens), dtype='float32')

# Doing one-hot representation using numpy
# This code will replace a specific character (token) by 1. which is present in the sentence
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestamp
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestamp and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

print(encoder_input_data[0].shape)

# Create LSTM Layer
# Define an input sequence and process it
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard "encoder_outputs" and only keep the states
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences, and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
          validation_split=0.2)

"""Inference model (sampling)
The drill:
1. encode input and retrieve initial decoder state
2. run one step of decoder with this initial state and a "start of sequence" token as target.
Output will be the next target token.
3. Repeat with the current target token and current states
"""

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_input_state_h = Input(shape=(latent_dim,))
decoder_input_state_c = Input(shape=(latent_dim,))
decoder_states_input = [decoder_input_state_h, decoder_input_state_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_input, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequence back to something readable
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_values = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_values)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_sequence_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_values = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sentence (part of the training set) fro trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decode_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decode_sentence)
