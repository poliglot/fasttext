from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation

DictionarySize = 10
HiddenSize     = 5
LearningRate   = 0.2
EmbeddingDim   = 100
MaxWords       = 20000
MaxSequenceLength = 100

texts = [
        "Hello world!",
        "How are you?"
        ]
labels = [0, 1]

tokenizer = Tokenizer(nb_words=MaxWords)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Map each word to its unique index
# {'you': 1, 'how': 2, 'hello': 3, 'are': 4, 'world': 5}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Zero-pad every string
data = pad_sequences(sequences, maxlen=MaxSequenceLength)

# TODO Why matrix?
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

model = Sequential()
# TODO MaxSequenceLength x DictionarySize
model.add(Dense(EmbeddingDim, input_dim=DictionarySize))
model.add(Activation('softmax'))
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
	vect[wordIndex] = 1
	return vect

def query(sentence):
	inputs    = tokenizer.texts_to_sequences([sentence])[0]
	iptOneHot = [oneHot(i) for i in inputs]
	return model.predict(np.asarray(iptOneHot))

print(query("world!"))
