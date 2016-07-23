#!/usr/bin/python
import time
import six.moves.cPickle

import model
import yelp_reader

model, tokeniser, dictionarySize = model.train(yelp_reader, oneHot = True, oneHotAveraged = True, contextHashes
 = False)

jsonModel = model.to_json()
open('model.json', 'w').write(jsonModel)
open('model-dictionary-size.dat', 'w').write(str(dictionarySize))
six.moves.cPickle.dump(tokeniser, open("tokeniser.pkl", "wb"))

model.save_weights('model-' + str(time.time()) + '.h5'
