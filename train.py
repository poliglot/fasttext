#!/usr/bin/python
import time
import six.moves.cPickle
from sklearn.manifold import TSNE
import numpy as np

import csv
import model as mdl
import yelp_reader

model, model2, tokeniser, dictionarySize = mdl.train(yelp_reader, oneHot = True, oneHotAveraged = True, contextHashes
 = True)

SamplesNum = 1000

testGenerator = mdl.mapGenerator(
  yelp_reader.validationData(SamplesNum + 100),
  tokeniser,
  dictionarySize,
  oneHot = True,
  oneHotAveraged = True,
  contextHashes = True
)

activations = model2.predict_generator((row[0] for row in testGenerator), val_samples=SamplesNum)

sampled   = [row for row in yelp_reader.validationData(SamplesNum + 100)][:SamplesNum]
sentences = [row[0] for row in sampled]
ratings   = [row[1] for row in sampled]

tsneModel  = TSNE(n_components=2, random_state=0)
tsneCoords = tsneModel.fit_transform(activations)

x = np.asarray([row[0] for row in tsneCoords])
y = np.asarray([row[1] for row in tsneCoords])

with open('data.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	writer.writerow(['x', 'y', 'rating', 'text'])
	for i, sentence in enumerate(sentences):
		writer.writerow([str(x[i]), str(y[i]), str(ratings[i]), sentence.replace("\n", " ")])

jsonModel = model.to_json()
open('model.json', 'w').write(jsonModel)
open('model-dictionary-size.dat', 'w').write(str(dictionarySize))
six.moves.cPickle.dump(tokeniser, open("tokeniser.pkl", "wb"))

model.save_weights('model-' + str(time.time()) + '.h5')