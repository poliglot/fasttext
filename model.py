from __future__ import print_function
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Merge

np.random.seed(1337)

EmbeddingDim    = 50
MaxWords        = 30000
SequenceLength  = 50
Epochs          = 5
SamplesPerEpoch = 1000
BatchSize       = 64
Labels          = 3
LabelMapping    = {
  1: 0,
  2: 0,
  3: 1,
  4: 2,
  5: 2
}

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
	t1 = sequence[t - 1] if t - 1 >= 0 else 0
	return (t1 * Prime1) % buckets

def triGramHash(sequence, t, buckets):
	t1 = sequence[t - 1] if t - 1 >= 0 else 0
	t2 = sequence[t - 2] if t - 2 >= 0 else 0

	return (t2 * Prime1 * Prime2 + t1 * Prime1) % buckets

def sentenceVector(tokeniser, dictionarySize, sentence, oneHotVectors, oneHotAveraged, contextHashes):
	result    = np.array([])
	sequences = tokeniser.texts_to_sequences([sentence])
	# Zero-pad every string
	padded    = pad_sequences(sequences, maxlen=SequenceLength)[0]

	if oneHotVectors:
		iptOneHot = [oneHot(dictionarySize, i) for i in padded]
		if oneHotAveraged:
			result = np.zeros(len(iptOneHot[0]))
			for iptOneHotVector in iptOneHot:
				result = result + iptOneHotVector

			result = result / len(iptOneHot)
		else:
			result = np.append(result, np.concatenate(iptOneHot))

	if contextHashes:
		buckets = np.zeros(dictionarySize * 2)
		for t in range(SequenceLength): buckets[biGramHash(padded, t, dictionarySize)] = 1
		for t in range(SequenceLength): buckets[dictionarySize + triGramHash(padded, t, dictionarySize)] = 1
		result = np.append(result, buckets)

	return result


def mapGenerator(generator, tokeniser, dictionarySize, oneHot, oneHotAveraged, contextHashes):
	for row in generator:
		sentence = row[0]
		label    = row[1]

		x = sentenceVector(tokeniser, dictionarySize, sentence, oneHot, oneHotAveraged, contextHashes)
		y = np.zeros(Labels)
		y[LabelMapping[label]] = 1
		yield (x[np.newaxis], y[np.newaxis])

def train(dataReader, oneHot, oneHotAveraged, contextHashes):
	n = (Epochs + 1) * SamplesPerEpoch  # TODO + 1 should not be needed

	tokeniser = Tokenizer(nb_words=MaxWords)
	tokeniser.fit_on_texts((row[0] for row in dataReader.trainingData(n)))

	# `word_index` maps each word to its unique index
	dictionarySize = len(tokeniser.word_index) + 1

	oneHotDimension        = (1 if oneHotAveraged else SequenceLength) * dictionarySize if oneHot else 0
	contextHashesDimension = dictionarySize * 2 if contextHashes else 0

	model = Sequential()
	model.add(Dense(EmbeddingDim, input_dim=(oneHotDimension + contextHashesDimension)))
	model.add(Dense(Labels, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	trainingGenerator   = mapGenerator(dataReader.trainingData(n),   tokeniser, dictionarySize, oneHot, oneHotAveraged, contextHashes)
	validationGenerator = mapGenerator(dataReader.validationData(n), tokeniser, dictionarySize, oneHot, oneHotAveraged, contextHashes)

	model.fit_generator(trainingGenerator,
		nb_epoch=Epochs,
		samples_per_epoch=SamplesPerEpoch,
		validation_data=validationGenerator,
		nb_val_samples=SamplesPerEpoch)

	return model, tokeniser, dictionarySize

# TODO Fix
def query(model, tokeniser, dictionarySize, sentence):
	concat = sentenceVector(tokeniser, dictionarySize, sentence)
	return model.predict(np.asarray(concat)[np.newaxis])
