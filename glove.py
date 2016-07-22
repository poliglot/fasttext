from __future__ import print_function
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Merge

np.random.seed(1337)

EmbeddingDim   = 50
MaxWords       = 30000
SequenceLength = 20
Labels         = 2
Epochs         = 5
BatchSize      = 64

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

# From https://primes.utm.edu/lists/small/100ktwins.txt
Prime1 = 15327749
Prime2 = 18409199

# `sequence` must refer to zero-padded sequence.
# From http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf, equation 6.6
def biGramHash(sequence, t, buckets):
	t1 = 0
	if (t - 1 >= 0): t1 = sequence[t - 1]

	return (t1 * Prime1) % buckets

def triGramHash(sequence, t, buckets):
	t1 = 0
	if (t - 1 >= 0): t1 = sequence[t - 1]

	t2 = 0
	if (t - 2 >= 0): t2 = sequence[t - 2]

	return (t2 * Prime1 * Prime2 + t1 * Prime1) % buckets

def sentenceVector(tokeniser, dictionarySize, sentence, oneHotVectors, contextHashes):
	result    = np.array([])
	sequences = tokeniser.texts_to_sequences([sentence])
	# Zero-pad every string
	padded    = pad_sequences(sequences, maxlen=SequenceLength)[0]

	if oneHotVectors:
		iptOneHot = [oneHot(dictionarySize, i) for i in padded]
		result    = np.append(result, np.concatenate(iptOneHot))

	if contextHashes:
		buckets = np.zeros(dictionarySize * 2)
		for t in range(SequenceLength): buckets[biGramHash(padded, t, dictionarySize)] = 1
		for t in range(SequenceLength): buckets[dictionarySize + triGramHash(padded, t, dictionarySize)] = 1
		result = np.append(result, buckets)

	return result

def train(x, y, oneHot, contextHashes):
	tokeniser = Tokenizer(nb_words=MaxWords)
	tokeniser.fit_on_texts(x)

	# Map each word to its unique index
	wordIndex      = tokeniser.word_index
	dictionarySize = len(wordIndex) + 1

	data   = [sentenceVector(tokeniser, dictionarySize, sentence, oneHot, contextHashes) for sentence in x]
	labels = [row for row in to_categorical(np.asarray(y))]

	print('Instances:', len(data))
	print('Dictionary size: ', dictionarySize)

	oneHotDimension = 0
	if oneHot: oneHotDimension = SequenceLength * dictionarySize

	contextHashesDimension = 0
	if contextHashes: contextHashesDimension = dictionarySize * 2

	model = Sequential()
	model.add(Dense(EmbeddingDim, input_dim=(oneHotDimension + contextHashesDimension)))
	model.add(Dense(Labels, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	x_train, y_train, x_val, y_val = prepare(data, labels)
	x_train = np.matrix(x_train)
	y_train = np.matrix(y_train)

	x_val = np.matrix(x_val)
	y_val = np.matrix(y_val)

	model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=Epochs, batch_size=BatchSize)

	return model, tokeniser, dictionarySize

def query(model, tokeniser, dictionarySize, sentence):
	concat = sentenceVector(tokeniser, dictionarySize, sentence)
	return model.predict(np.asarray(concat)[np.newaxis])
