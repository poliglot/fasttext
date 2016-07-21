from __future__ import print_function
import os
import numpy as np
import data_reader
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Merge

DictionarySize = 5277  # V
HiddenSize1    = 50   # Hidden layer for sequence elements
HiddenSize2    = 10   # Hidden layer after sequence elements have been merged
LabelsLength   = 2
LearningRate   = 0.2
EmbeddingDim   = 300
MaxWords       = 20000
SequenceLength = 10

dataset = data_reader.dataset()
texts = list(map(lambda x: x[1], dataset))

labels = [0, 1]

tokenizer = Tokenizer(nb_words=MaxWords)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(sequences)

# Map each word to its unique index
# {'you': 1, 'how': 2, 'hello': 3, 'are': 4, 'world': 5}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Zero-pad every string
data = pad_sequences(sequences, maxlen=SequenceLength)

print(data)

# TODO Why matrix?
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

model = Sequential()
model.add(Dense(EmbeddingDim, input_dim=SequenceLength * DictionarySize))
model.add(Dense(LabelsLength, activation='softmax'))
model.compile(optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

#model.fit(x_train, y_train, validation_data=(x_val, y_val),
#          nb_epoch=2, batch_size=128)

# Derive n-grams
def nGram(n, s):
    return []

def oneHot(wordIndex):
	vect = np.zeros(len(word_index) + 1)
	if wordIndex > 0:
		vect[wordIndex] = 1
	return vect

def query(sentence):
	_sequences = tokenizer.texts_to_sequences([sentence])
	_padded = pad_sequences(_sequences, maxlen=SequenceLength)[0]
	iptOneHot = [oneHot(i) for i in _padded]
	concat = np.concatenate(iptOneHot)[np.newaxis]
	return model.predict(concat)

print(query("It is"))
