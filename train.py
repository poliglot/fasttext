#!/usr/bin/python
import time
import six.moves.cPickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

import model as mdl
import yelp_reader

model, model2, tokeniser, dictionarySize = mdl.train(yelp_reader, oneHot = True, oneHotAveraged = True, contextHashes
 = True)

testGenerator = mdl.mapGenerator(
  yelp_reader.trainingData(500),
  tokeniser,
  dictionarySize,
  oneHot = True,
  oneHotAveraged = True,
  contextHashes = True
)

activations = model2.predict_generator((row[0] for row in testGenerator), val_samples=100)
sentences = [row[0] for row in yelp_reader.trainingData(500)][:100]

# print(sentences)

tsne_model = TSNE(n_components=2, random_state=0)
x_y = tsne_model.fit_transform(activations)

plt.scatter(np.asarray([row[0] for row in x_y]), np.asarray([row[1] for row in x_y]), cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
# plt.clim(-100.0, 100)
plt.show()

jsonModel = model.to_json()
open('model.json', 'w').write(jsonModel)
open('model-dictionary-size.dat', 'w').write(str(dictionarySize))
six.moves.cPickle.dump(tokeniser, open("tokeniser.pkl", "wb"))

model.save_weights('model-' + str(time.time()) + '.h5')

