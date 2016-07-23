import glob
import six.moves.cPickle
from keras.models import model_from_json

import model

tokeniser = six.moves.cPickle.load(open("tokeniser.pkl", 'rb'))

lastModel = sorted(glob.glob('model-*.h5'))[-1]

model = model_from_json(open("model.json").read())
model.load_weights(lastModel)

dictionarySize = int(open('model-dictionary-size.dat').read())

print(model.query(model, tokeniser, dictionarySize, "It is bad"))
print(model.query(model, tokeniser, dictionarySize, "It is good"))
