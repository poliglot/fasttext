from __future__ import print_function
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Merge

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
	concat    = np.concatenate(iptOneHot)
	return concat

def save(model):
	json_string = model.to_json()
	model.save_weights('my_model_weights.h5')

def train(x, y):
	tokeniser = Tokenizer(nb_words=MaxWords)
	tokeniser.fit_on_texts(x)

	# Map each word to its unique index
	wordIndex      = tokeniser.word_index
	dictionarySize = len(wordIndex) + 1

	data   = [sentenceVector(tokeniser, dictionarySize, sentence) for sentence in x]
	labels = [row for row in to_categorical(np.asarray(y))]

	print('Instances:', len(data))
	print('Dictionary size: ', dictionarySize)

	model = Sequential()
	model.add(Dense(EmbeddingDim, input_dim=SequenceLength * dictionarySize))
	model.add(Dense(Labels, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	x_train, y_train, x_val, y_val = prepare(data, labels)
	x_train = np.matrix(x_train)
	y_train = np.matrix(y_train)

	x_val = np.matrix(x_val)
	y_val = np.matrix(y_val)

	model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=2, batch_size=128)

	return model, tokeniser, dictionarySize

def query(model, tokeniser, dictionarySize, sentence):
	concat = sentenceVector(tokeniser, dictionarySize, sentence)
	return model.predict(np.asarray(concat)[np.newaxis])