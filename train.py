#!/usr/bin/python
import time
import six.moves.cPickle

import glove
# import data_reader
import yelp_reader

model, tokeniser, dictionarySize = glove.train(yelp_reader, oneHot = False, contextHashes = True)

jsonModel = model.to_json()
open('model.json', 'w').write(jsonModel)
open('model-dictionary-size.dat', 'w').write(str(dictionarySize))
six.moves.cPickle.dump(tokeniser, open("tokeniser.pkl", "wb"))

model.save_weights('model-' + str(time.time()) + '.h5')
