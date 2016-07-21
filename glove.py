from __future__ import print_function
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Merge

import data_reader

np.random.seed(1337)

HiddenSize     = 10
EmbeddingDim   = 300
MaxWords       = 20000
SequenceLength = 10
Labels         = 2

def prepare(data, labels):
	VALIDATION_SPLIT = 0.2
	nb_validation_samples = int(VALIDATION_SPLIT * len(data))

	x_train = data[:-nb_validation_samples]
	y_train = labels[:-nb_validation_samples]
	x_val   = data[-nb_validation_samples:]
	y_val   = labels[-nb_validation_samples:]

	return x_train, y_train, x_val, y_val

def oneHot(dictionarySize, wordIndex):
	vect = np.zeros(dictionarySize)
	if wordIndex > 0: vect[wordIndex] = 1
	return vect

def sentenceVector(tokeniser, dictionarySize, sentence):
	sequences = tokeniser.texts_to_sequences([sentence])
	# Zero-pad every string
	padded    = pad_sequences(sequences, maxlen=SequenceLength)[0]
	iptOneHot = [oneHot(dictionarySize, i) for i in padded]
	concat    = np.concatenate(iptOneHot)[np.newaxis]
	return concat

def train(x, y):
	tokeniser = Tokenizer(nb_words=MaxWords)
	tokeniser.fit_on_texts(x)

	# Map each word to its unique index
	word_index = tokeniser.word_index

	dictionarySize = len(word_index) + 1

	data = list(map(lambda sentence: sentenceVector(tokeniser, dictionarySize, sentence), x))

	# sequences = tokeniser.texts_to_sequences(x)
	# data = pad_sequences(sequences, maxlen=SequenceLength)

	labels = [row[np.newaxis] for row in to_categorical(np.asarray(y))]

#	print('Shape of data tensor:', x.shape)
#	print('Shape of label tensor:', labels.shape)
#	print('Dictionary size: ', dictionarySize)

	model = Sequential()
	model.add(Dense(EmbeddingDim, input_dim=SequenceLength * dictionarySize))
	model.add(Dense(Labels, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	x_train, y_train, x_val, y_val = prepare(data, labels)

	model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=2, batch_size=128)

	return model, tokeniser, dictionarySize

def query(model, tokeniser, dictionarySize, sentence):
	return model.predict(concat)

dataset = data_reader.dataset()
x = list(map(lambda x: x[1], dataset))[0:100]
y = list(map(lambda x: x[0], dataset))[0:100]
model, tokeniser, dictionarySize = train(x, y)

print(query(model, tokeniser, dictionarySize, "It is bad"))
print(query(model, tokeniser, dictionarySize, "It is good"))
